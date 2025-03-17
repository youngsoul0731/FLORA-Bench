# -*- coding: utf-8 -*-
# @Date    :
# @Author  : all
# @Desc    : test on gsm8k
import re
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger

from scripts.utils.extract_GSM8K_workflow import CallGraphParser
from scripts.utils.fix_extarct_workflow import fix_test_extract_workflow
from datetime import datetime
from scripts.pgy_dataset import CustomGraphDataset
from torch_geometric.loader import DataLoader
import torch


class GSM8KBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_number(self, text: str) -> Optional[float]:
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        else:
            return None

    def calculate_score(self, expected_output: float, prediction: float) -> Tuple[float, float]:
        if prediction is None:
            return 0.0, prediction
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, float, float, float]:
        input_text = problem["question"]
        expected_output = self.extract_number(problem["answer"])

        try:
            output, cost = await self._generate_output(graph, input_text)
            predicted_number = self.extract_number(output)
            score, extracted_output = self.calculate_score(expected_output, predicted_number)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]

    async def gnn_evaluate_problem(self, data: List[dict], graph: Callable, 
                                   device = "cpu", model = None, encoder = None, memory = None, random = False) -> Tuple[str, str]:
        # unstable extraction
        nodes, edges = self.extract_workflow(graph)
        
        raw_data_list = []
        for problem in data:
            item = {
                "task":problem["question"],
                "nodes": nodes,
                "edges": edges
            }
            raw_data_list.append(item)

        raw_data_list = self.process_raw_data_list(raw_data_list)
        dataset = CustomGraphDataset(raw_data_list = raw_data_list, encoder = encoder, memory = memory)
        loader = DataLoader(dataset.data_list, batch_size=len(dataset.data_list), shuffle=False)

        
        try:
            scores = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    num_graphs = batch.batch[-1] + 1    
                    batch.task_embedding = batch.task_embedding.reshape(num_graphs, -1)
                    if random:
                        score = torch.rand(len(batch.task_embedding)).to(device)
                    else:
                        score = model(batch.x, batch.task_embedding, batch.edge_index, batch.batch)
                    preds = (score > 0.5).float()
                    preds_list = preds.view(-1).tolist()
                    scores.extend(preds_list)
            
            output = []
            for i, problem in enumerate(data):
                

                update_item =[
                    problem["question"], "", problem["answer"], scores[i], 0.0
                ]
                output.append(update_item)
            # return batch's output
            return output
            # self.log_mismatch(input_text, expected_output, output, extracted_output)
            # return input_text, output, extracted_output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            print(e)
            return "", str(e), str(e), "", 0.0, 0.0
    

    def extract_workflow(self, graph):
        nodes = []
        edges = []
        # TODO: extract nodes and edges from graph
        
        parser = CallGraphParser()
        graph_cls = type(graph)
        parser.parse_obj(graph_cls)
        
        nodes, edges = parser.extract_graph_data()
        nodes, edges = fix_test_extract_workflow(nodes, edges, graph_cls)

        return nodes, edges
    


    def process_raw_data_list(self, raw_data_list):
        output_data_list = []
        for item in raw_data_list:
            item['source'] = 'mmlu_aflow'   
            item['model'] = 'gpt-4o-mini'   
            node_info = {}
            item['nodes'][0]['prompt'] = "This workflow starts with task initialization"
            for node in item['nodes']:
                node_info[node['name']] = node['prompt']
            node_id_map = {k:i for i,k in enumerate(node_info.keys())}  
            temp_node_info = {node_id_map[k]:v for k,v in node_info.items()}
            item['nodes'] = temp_node_info
            temp_edge_index = [[node_id_map[e['input']],node_id_map[e['output']]] for e in item['edges']]    
            item['edge_index'] = temp_edge_index
            keys_to_keep = ['nodes','edge_index','task','source','model']  
            item = {k:item[k] for k in keys_to_keep}    
            output_data_list.append(item)  
        
        return output_data_list