import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger
from metagpt.utils.sanitize import sanitize

from scripts.utils.extract_MBPP_workflow import CallGraphParser
from scripts.utils.fix_extarct_workflow import fix_test_extract_workflow
from datetime import datetime
from scripts.pgy_dataset import CustomGraphDataset
from torch_geometric.loader import DataLoader
import torch


class MBPPBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    class TimeoutError(Exception):
        pass

    def run_with_timeout(self, func, timeout):
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func())
            except Exception as e:
                result.append(e)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def check_solution(self, solution, test, entry_point):
        solution = sanitize(code=solution, entrypoint=entry_point)
        try:
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            exec(solution, global_dict)

            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} is not defined in the solution.")

            exec(test, global_dict)

            check = global_dict["check"]

            result = self.run_with_timeout(check, 15)

            if result is None:
                result = (self.PASS, "The solution passed all test cases.")

        except self.TimeoutError:
            result = (
                self.FAIL,
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
            )
        except Exception as e:
            error_message = f"Error: {str(e)}.\n Solution: {solution}.\n Test: {test}"
            result = (self.FAIL, error_message)

            with open("error.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_message}\n")

        return result

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, prompt, entry_point):
        return await graph(prompt, entry_point)

    async def evaluate_problem(self, data: dict, graph: Callable) -> Tuple[str, str, str, float, float]:
        input_text = data["prompt"]
        expected_output = "\nCorrect Solution:\ndef " + data["code"]

        try:
            # Generate prediction using the graph function
            prediction, cost = await self._generate_output(graph, input_text, data["entry_point"])

            # Check the solution
            ret = self.check_solution(prediction, data["test"], data["entry_point"])
            test_case_details = ret[1]
            expected_output = test_case_details + "\nCorrect Solution:" + data["code"]

            # Calculate score based on the check result
            score = 1.0 if ret[0] == self.PASS else 0.0

            # Log mismatch if the score is 0
            if score == 0:
                self.log_mismatch(input_text, expected_output, prediction, score)

            return input_text, prediction, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[float, str]:
        # The scoring logic for MBPP is already implemented in evaluate_problem, this is just to conform to the interface
        return 0.0, prediction

    def get_result_columns(self) -> List[str]:
        return ["inputs", "prediction", "expected_output", "score", "cost"]
    
    async def gnn_evaluate_problem(self, data: List[dict], graph: Callable, 
                                   device = "cpu", model = None, encoder = None, memory = None, random = False) -> Tuple[str, str]:
        # unstable extraction
        nodes, edges = self.extract_workflow(graph)
        
        raw_data_list = []
        for problem in data:
            item = {
                "task":problem["prompt"],
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
                    problem["prompt"], "", "", scores[i], 0.0
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