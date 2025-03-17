import inspect
import re
from math import isclose
from typing import Any, Callable, List, Tuple

import regex
from sympy import N, simplify
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger

from scripts.utils.extract_MATH_workflow import CallGraphParser
from scripts.utils.fix_extarct_workflow import fix_test_extract_workflow
from datetime import datetime
from scripts.pgy_dataset import CustomGraphDataset
from torch_geometric.loader import DataLoader
import torch


class MATHBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_model_answer(self, text: str) -> str:
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        sentence_end_pattern = r"(?<!\d)[.!?]\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        else:
            return 0, predicted_answer

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        if str(prediction) == str(reference):
            return True

        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except:
            pass

        try:
            return self.symbolic_equal(prediction, reference)
        except:
            pass

        return False

    def is_digit(self, num):
        return self.parse_digits(num) is not None

    def parse_digits(self, num):
        num = regex.sub(",", "", str(num))
        try:
            return float(num)
        except:
            if num.endswith("%"):
                num = num[:-1]
                if num.endswith("\\"):
                    num = num[:-1]
                try:
                    return float(num) / 100
                except:
                    pass
        return None

    def symbolic_equal(self, a, b):
        def _parse(s):
            for f in [parse_latex, parse_expr]:
                try:
                    return f(s)
                except:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        try:
            if simplify(a - b) == 0:
                return True
        except:
            pass

        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except:
            pass
        return False

    def get_function_code(self, func):
        try:
            source_code = inspect.getsource(func)
            return source_code
        except OSError:
            return "no code"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, int, float]:
        input_text = problem["problem"]
        expected_output = problem["solution"]

        try:
            output, cost = await self._generate_output(graph, input_text)
            uni_score, extracted_output = self.calculate_score(expected_output, output)

            
            self.log_mismatch(
                input_text,
                expected_output,
                output,
                extracted_output,
                extract_answer_code=self.get_function_code(self.extract_model_answer),
            )

            return input_text, output, expected_output, uni_score, cost

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
                "task":problem["problem"],
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
                    problem["problem"], "", problem["solution"],  scores[i], 0.0
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