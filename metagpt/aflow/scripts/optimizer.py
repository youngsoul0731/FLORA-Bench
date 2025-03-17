# -*- coding: utf-8 -*-
# @Date    : 8/12/2024 22:00 PM
# @Author  : issac
# @Desc    : optimizer for graph

import asyncio
import os
import time
import random
from typing import List, Literal

from pydantic import BaseModel, Field

from metagpt.actions.action_node import ActionNode
from metagpt.aflow.scripts.evaluator import DatasetType
from metagpt.aflow.scripts.optimizer_utils.convergence_utils import ConvergenceUtils
from metagpt.aflow.scripts.optimizer_utils.data_utils import DataUtils
from metagpt.aflow.scripts.optimizer_utils.evaluation_utils import EvaluationUtils
from metagpt.aflow.scripts.optimizer_utils.experience_utils import ExperienceUtils
from metagpt.aflow.scripts.optimizer_utils.graph_utils import GraphUtils
from metagpt.utils.cost_manager import CostManager
from metagpt.logs import logger
from metagpt.provider.llm_provider_registry import create_llm_instance
from benchmark.benchmark import BaseBenchmark
from benchmark.drop import DROPBenchmark
from benchmark.gsm8k import GSM8KBenchmark
from benchmark.hotpotqa import HotpotQABenchmark
from benchmark.humaneval import HumanEvalBenchmark
from benchmark.math import MATHBenchmark
from benchmark.mbpp import MBPPBenchmark
from benchmark.mmlu import MMLUBenchmark
from typing import Dict, Literal, Tuple
import aiofiles
import json

QuestionType = Literal["math", "code", "qa"]
OptimizerType = Literal["Graph", "Test"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data(json_file_path):
    data = []
    with open(json_file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


class GraphOptimize(BaseModel):
    modification: str = Field(default="", description="modification")
    graph: str = Field(default="", description="graph")
    prompt: str = Field(default="", description="prompt")


class Optimizer:
    def __init__(
        self,
        dataset: DatasetType,
        question_type: QuestionType,
        opt_llm_config,
        exec_llm_config,
        operators: List,
        sample: int,
        check_convergence: bool = False,
        optimized_path: str = None,
        initial_round: int = 1,
        max_rounds: int = 20,
        validation_rounds: int = 5,
        workflow_dir: str = None,
        labels_dir: str = None,
        random_rate: float = 0.1,
        gnn_optimize: bool=False,
        model_name: str = None,
        ckp_base_path: str = None,
        dataset_file: str = None,
        random_test: bool = False
       
    ) -> None:
        self.optimize_llm_config = opt_llm_config
        self.optimize_llm = create_llm_instance(self.optimize_llm_config)
        self.execute_llm_config = exec_llm_config

        self.dataset = dataset
        self.type = question_type
        self.check_convergence = check_convergence

        self.graph = None
        self.operators = operators

        self.root_path = f"{optimized_path}/{self.dataset}"
        self.sample = sample
        self.top_scores = []
        self.round = initial_round
        self.max_rounds = max_rounds
        self.validation_rounds = validation_rounds

        self.graph_utils = GraphUtils(self.root_path)
        self.data_utils = DataUtils(self.root_path)
        self.experience_utils = ExperienceUtils(self.root_path)
        self.evaluation_utils = EvaluationUtils(self.root_path)
        self.convergence_utils = ConvergenceUtils(self.root_path)
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
            "HotpotQA": HotpotQABenchmark,
            "MBPP": MBPPBenchmark,
            "DROP": DROPBenchmark,
            "MMLU": MMLUBenchmark,
        }
        self.benchmark_class = self.dataset_configs[self.dataset]
        
        self.graph_dir = f"{self.root_path}/workflows" if workflow_dir is None else workflow_dir
        self.labels_dir = f"{self.root_path}/results" if labels_dir is None else labels_dir
        self.random_rate = random_rate
        self.gnn_optimize = gnn_optimize
        self.model_name = model_name
        self.ckp = ckp_base_path
        self.memory = None
        self.dataset_file = dataset_file
        self.random_test = random_test
        

    async def get_dataset_len(self, dataset: DatasetType, test: bool) -> int:
        base_path = f"data/{dataset.lower()}"
        file_path = f"{base_path}_test.jsonl" if test else f"{base_path}_validate.jsonl"
        data = []
        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        return len(data)

    
    def optimize(self, mode: OptimizerType = "Graph"):
        if mode == "Test":
            test_n = 3  # validation datasets's execution number
            for i in range(test_n):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                score = loop.run_until_complete(self.test(self.random_test))
            return None

            
        len_dataset = asyncio.run(self.get_dataset_len(self.dataset, test=False))
        num_rounds = min(self.max_rounds, len_dataset)
        print(f"Number of rounds: {num_rounds}-----------------------------------")
        if self.gnn_optimize:
            self.memory = asyncio.run(self.evaluation_utils.load_dataset_embedding(self.dataset))
        
        for opt_round in range(num_rounds):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            retry_count = 0
            max_retries = 1

            while retry_count < max_retries:
                try:
                    score = loop.run_until_complete(self._optimize_graph())
                    break
                except Exception as e:
                    retry_count += 1
                    logger.info(f"Error occurred: {e}. Retrying... (Attempt {retry_count}/{max_retries})")
                    if retry_count == max_retries:
                        logger.info("Max retries reached. Moving to next round.")
                        score = None

                    wait_time = 5 * retry_count
                    time.sleep(wait_time)

                if retry_count < max_retries:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

            self.round += 1
            logger.info(f"Score for round {self.round}: {score}")

            converged, convergence_round, final_round = self.convergence_utils.check_convergence(top_k=3)

            if converged and self.check_convergence:
                logger.info(
                    f"Convergence detected, occurred in round {convergence_round}, final round is {final_round}"
                )
                # Print average scores and standard deviations for each round
                self.convergence_utils.print_results()
                break

            time.sleep(5)

    async def _optimize_graph(self):
        validation_n = self.validation_rounds  # validation datasets's execution number
        graph_path = f"{self.root_path}/workflows"
        data = self.data_utils.load_results(graph_path)
        # va_list = [self.round, self.round + 1, self.round + 2, self.round + 3, self.round + 4]
        # va_list = [self.round]
        va_list = None
        if self.round == 1:
            directory = self.graph_utils.create_round_directory(graph_path, self.round)
            # Load graph using graph_utils
            self.graph = self.graph_utils.load_graph(self.round, graph_path)
            avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=True, va_list=va_list, 
                                                                   gnn_optimize=self.gnn_optimize, model_name=self.model_name, ckp=self.ckp, memory=self.memory)
            
        cost_manager = CostManager()
        # Create a loop until the generated graph meets the check conditions
        while True:
            should_break = False
            directory = self.graph_utils.create_round_directory(graph_path, self.round + 1)
            top_rounds = self.data_utils.get_top_rounds(self.sample)
            all_rounds = self.data_utils.get_all_rounds()
            # self.sample_rate randomly sample item from all_rounds
            if random.random() < self.random_rate:
                sample = random.choice(all_rounds)
            else:   
                sample = self.data_utils.select_round(top_rounds)

            for i in range(5):
                prompt, graph_load = self.graph_utils.read_graph_files(sample["round"], graph_path)
                graph = self.graph_utils.extract_solve_graph(graph_load)

                processed_experience = self.experience_utils.load_experience()
                experience = self.experience_utils.format_experience(processed_experience, sample["round"])

                operator_description = self.graph_utils.load_operators_description(self.operators)
                log_data = self.data_utils.load_log(sample["round"])

                graph_optimize_prompt = self.graph_utils.create_graph_optimize_prompt(
                    experience, sample["score"], graph[0], prompt, operator_description, self.type, log_data
                )

                graph_optimize_node = await ActionNode.from_pydantic(GraphOptimize).fill(
                    context=graph_optimize_prompt, mode="xml_fill", llm=self.optimize_llm
                )

                response = await self.graph_utils.get_graph_optimize_response(graph_optimize_node)

                # Check if the modification meets the conditions
                check = self.experience_utils.check_modification(
                    processed_experience, response["modification"], sample["round"]
                )

                # If `check` is True, break the loop; otherwise, regenerate the graph
                if check:
                    should_break = True
                    break
            if should_break:
                break

        cost = cost_manager.total_cost
        # Save the graph and evaluate
        self.graph_utils.write_graph_files(directory, response, self.round + 1, self.dataset)

        experience = self.experience_utils.create_experience_data(sample, response["modification"])

        self.graph = self.graph_utils.load_graph(self.round + 1, graph_path)

        logger.info(directory)
        
        avg_score = await self.evaluation_utils.evaluate_graph(self, directory, validation_n, data, initial=False, va_list = va_list, 
                                                               gnn_optimize=self.gnn_optimize, model_name=self.model_name, ckp=self.ckp, memory=self.memory, cost=cost)

        self.experience_utils.update_experience(directory, experience, avg_score)

        return avg_score


    def load_data(self, json_file_path):
        data = []
        with open(json_file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    async def test(self, random_test = False):
        

        graph_path = self.graph_dir
        save_path = self.labels_dir
        
        # go through subdirs under graph_path and return workflow_paths list
        workflow_paths = [os.path.join(graph_path, subdir) for subdir in os.listdir(graph_path) if "round" in subdir]
        if random_test:
            workflow_paths = random.sample(workflow_paths, 3)
        print(f"Workflow paths: {workflow_paths}")
        save_file_path = self.data_utils.get_eval_file_path(graph_path)

        data = []

        for _, workflow_path in enumerate(workflow_paths, 1):
            round = int(workflow_path.split("/")[-1].split("_")[-1])
            directory = self.graph_utils.create_round_directory(save_path, round)
            # self.graph = self.graph_utils.load_graph(round, graph_path)
            self.graph = self.graph_utils.load_graph_from_particular_file(workflow_path)

            score, avg_cost, total_cost = await self.evaluation_utils.evaluate_graph_test(self, directory, is_test=True, dataset_file = self.dataset_file)
            new_data = self.data_utils.create_result_data(round, score, avg_cost, total_cost)
            data.append(new_data)

            self.data_utils.save_results(save_file_path, data)
