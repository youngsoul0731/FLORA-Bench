# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 10:00 AM
# @Author  : all
# @Desc    : Evaluation for different datasets

from typing import Dict, Literal, Tuple, List
import json
import torch
import aiofiles
from tqdm import tqdm
from benchmark.benchmark import BaseBenchmark
from benchmark.drop import DROPBenchmark
from benchmark.gsm8k import GSM8KBenchmark
from benchmark.hotpotqa import HotpotQABenchmark
from benchmark.humaneval import HumanEvalBenchmark
from benchmark.math import MATHBenchmark
from benchmark.mbpp import MBPPBenchmark
from benchmark.mmlu import MMLUBenchmark
# If you want to customize tasks, add task types here and provide evaluation functions, just like the ones given above
DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP", "MMLU"]
TASK_DICT = {
            "MMLU": "task",
            "HumanEval": "prompt",
            "GSM8K": "question",
            "MATH": "problem",
            "MBPP": "prompt"
        }
from scripts.utils.embedding import get_encoder
from scripts.utils.model import get_model
import torch

class Evaluator:
    """
    Complete the evaluation for different datasets here
    """

    def __init__(self, eval_path: str):
        self.eval_path = eval_path
        self.dataset_configs: Dict[DatasetType, BaseBenchmark] = {
            "GSM8K": GSM8KBenchmark,
            "MATH": MATHBenchmark,
            "HumanEval": HumanEvalBenchmark,
            "HotpotQA": HotpotQABenchmark,
            "MBPP": MBPPBenchmark,
            "DROP": DROPBenchmark,
            "MMLU": MMLUBenchmark,
        }

    async def graph_evaluate(
        self, dataset: DatasetType, graph, params: dict, path: str, is_test: bool = False, va_list: list = None, 
        gnn_optimize = None, model_name = None, ckp = None, memory = None, dataset_file = None
    ) -> Tuple[float, float, float]:
        if dataset not in self.dataset_configs:
            raise ValueError(f"Unsupported dataset: {dataset}")
        if dataset_file is not None:
            data_path = dataset_file
        else:
            data_path = self._get_data_path(dataset, is_test)
        
        benchmark_class = self.dataset_configs[dataset]
        benchmark = benchmark_class(name=dataset, file_path=data_path, log_path=path)

        # Use params to configure the graph and benchmark
        
        configured_graph = await self._configure_graph(dataset, graph, params)
        
        if is_test:
            va_list = None  # For test data, generally use None to test all
        else:
            va_list = va_list  # Use None to test all Validation data, or set va_list (e.g., [1, 2, 3]) to use partial data
        return await benchmark.run_evaluation(graph = configured_graph, va_list = va_list, 
                                              gnn_optimize = gnn_optimize, model_name = model_name, ckp = ckp, memory = memory)

    async def _configure_graph(self, dataset, graph, params: dict):
        # Here you can configure the graph based on params
        # For example: set LLM configuration, dataset configuration, etc.
        dataset_config = params.get("dataset", {})
        llm_config = params.get("llm_config", {})
        return graph(name=dataset, llm_config=llm_config, dataset=dataset_config)

    def _get_data_path(self, dataset: DatasetType, test: bool) -> str:
        base_path = f"data/{dataset.lower()}"
        return f"{base_path}_test.jsonl" if test else f"{base_path}_validate.jsonl"
    
    async def load_data(self, file_path, specific_indices: List[int] = None) -> List[dict]:
        data = []
        async with aiofiles.open(file_path, mode="r", encoding="utf-8") as file:
            async for line in file:
                data.append(json.loads(line))
        if specific_indices is not None:
            filtered_data = [data[i] for i in specific_indices if i < len(data)]
            return filtered_data
        return data
    
    async def load_dataset_embedding(self, dataset: DatasetType, is_test: bool = False):
        # Load dataset embeddings if needed
        data_path = self._get_data_path(dataset, is_test)
        dataset_list = await self.load_data(data_path)
        index = TASK_DICT[dataset]

        memory = {}
        encoder = get_encoder('ST', cache_dir="./model", batch_size=1)
        for data in tqdm(dataset_list,desc="Getting prompt embedding"):
            if data[index] not in memory.keys():
                memory[data[index]] = None 
        task_attr = list(memory.keys())
        task_attr_embedding = encoder.encode(task_attr)
        for attr in task_attr:
            memory[attr] = task_attr_embedding[task_attr.index(attr)]   
        print("dataset embedding done")
        return memory

            
