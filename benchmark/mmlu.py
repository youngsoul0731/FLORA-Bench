from metagpt.logs import logger
import glob
import pandas as pd
from typing import Union, List, Literal, Any, Dict, Callable,  Tuple
import re
import string
from collections import Counter
from benchmark.benchmark import BaseBenchmark

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import numpy as np
from abc import ABC

from scripts.utils.extract_MMLU_workflow import CallGraphParser
from scripts.utils.fix_extarct_workflow import fix_test_extract_workflow
from datetime import datetime
from scripts.pgy_dataset import CustomGraphDataset
from torch_geometric.loader import DataLoader
import torch



class MMLUBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.workflows_file = None

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) == 1:
            return answer
        if answer.startswith("Option"):
            return answer[6]
        if len(answer) > 0:
            ans_pos = answer.find("answer is")
            if ans_pos != -1:
                extracted = answer[ans_pos + len("answer is"):].strip(":").strip().strip("*").strip().strip("Option").strip()
                match = re.search(r'([A-Z])', extracted, re.IGNORECASE)
                if match:
                    return match.group(1)
                answer = answer[ans_pos+len("answer is"):].strip(":").strip().strip("*").strip().strip("Option").strip()
            answer = answer[0] # Try to format the answer by taking the first letter
        return answer
    
    
    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        prediction = self.postprocess_answer(prediction)
        if prediction == expected_output:
            return 1.0, prediction
        else:
            return 0.0, prediction
        
    
    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)
    
    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str]:
        input_text = problem["task"] 
        expected_output = problem["answer"]
        
        try:

            output, cost = await self._generate_output(graph, input_text)
            score, extracted_output = self.calculate_score(expected_output, output)
            self.log_mismatch(input_text, expected_output, output, extracted_output)
            
            return input_text, output, extracted_output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            print(e)
            return input_text, str(e), str(e), expected_output, 0.0, 0.0
        
    async def gnn_evaluate_problem(self, data: List[dict], graph: Callable, 
                                   device = "cpu", model = None, encoder = None, memory = None, random = False) -> Tuple[str, str]:
        # unstable extraction
        nodes, edges = self.extract_workflow(graph)
        
        raw_data_list = []
        for problem in data:
            item = {
                "task":problem["task"],
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
                    problem["task"], "", "", problem["answer"], scores[i], 0.0
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
    
    def get_result_columns(self) -> List[str]:
        return ["input_text", "output_text", "prediction", "answer", "score", "cost"]
    
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