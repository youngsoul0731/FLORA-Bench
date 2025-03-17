# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 20:00 PM
# @Author  : didi
# @Desc    : Entrance of AFlow.

import argparse
from typing import Dict, List

import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from metagpt.configs.models_config import ModelsConfig
from data.download_data import download
from metagpt.aflow.scripts.optimizer import Optimizer

import os
class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators


EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "DROP": ExperimentConfig(
        dataset="DROP",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
    "HotpotQA": ExperimentConfig(
        dataset="HotpotQA",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
    ),
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K",
        question_type="math",
        operators=["Custom", "ScEnsemble", "Programmer"],
    ),
    "MBPP": ExperimentConfig(
        dataset="MBPP",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        operators=["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"],
    ),
    "MMLU": ExperimentConfig(
        dataset="MMLU",
        question_type="qa",
        operators=["Custom", "AnswerGenerate", "ScEnsemble"],
        ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="AFlow Optimizer")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        # required=True,
        help="Dataset type",
        default="HumanEval"
    )
    parser.add_argument("--sample", type=int, default=4, help="Sample count")
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="updated",
        help="Optimized result save path",
    )
    parser.add_argument("--initial_round", type=int, default=3, help="Initial round")
    parser.add_argument("--max_rounds", type=int, default=200, help="Max iteration rounds")
    parser.add_argument("--check_convergence", type=bool, default=False, help="Whether to enable early stop")
    parser.add_argument("--validation_rounds", type=int, default=3, help="Validation rounds")
    parser.add_argument("--llm_config", type=str, default="gpt-4o-mini", help="LLM config")
    parser.add_argument(
        "--if_first_optimize",
        type=lambda x: x.lower() == "true",
        default = False,
        help="Whether to download dataset for the first time",
    )
    parser.add_argument("--gnn_optimize", type=bool, default=False, help="Whether to use GNN to optimize")
    parser.add_argument("--workflow_dir", type=str, nargs="?",default="experiments/HumanEval/labels_latest", help="Workflow directory")
    parser.add_argument("--labels_dir", type=str, nargs="?",default="experiments/HumanEval/deepseek_labels", help="Labels directory")
    parser.add_argument("--random_rate", type=float, default=0.3, help="Random rate")
    parser.add_argument("--dataset_file", type=str, default=None, help="Dataset file")
    parser.add_argument("--random_test", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    download(["datasets"], if_first_download=args.if_first_optimize)
    config = EXPERIMENT_CONFIGS[args.dataset]
    mini_llm_config =  ModelsConfig.default().get(args.llm_config)
    deepseek_llm_config = ModelsConfig.default().get("deepseek-chat")
    # claude_llm_config = ModelsConfig.default().get("claude-3-5-sonnet-20240620")

    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=mini_llm_config,
        exec_llm_config=mini_llm_config,
        check_convergence=args.check_convergence,
        operators=config.operators,
        optimized_path=args.optimized_path,
        sample=args.sample,
        initial_round=args.initial_round,
        max_rounds=args.max_rounds,
        validation_rounds=args.validation_rounds,
        workflow_dir=args.workflow_dir,
        labels_dir=args.labels_dir,
        random_rate=args.random_rate,
        gnn_optimize=args.gnn_optimize,
        random_test=args.random_test
    )

    # Optimize workflow via setting the optimizer's mode to 'Graph'
    # optimizer.optimize("Graph")

    # Test workflow via setting the optimizer's mode to 'Test'
    optimizer.optimize("Test")
