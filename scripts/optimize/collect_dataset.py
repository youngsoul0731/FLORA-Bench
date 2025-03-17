import os
import json
import argparse
import shutil
import time
from datetime import datetime
from utils.fix_predicitions import fix_score
from utils.get_dataset_len import get_dataset_len, get_dataset
def get_nodes_prompt(node_name):
    if ("custom" in node_name.lower()) and ("customcodegenerate" not in node_name.lower()):
        return "Custom"
    elif "answergenerate" in node_name.lower():
        return """
    Think step by step and solve the problem.
    1. In the "thought" field, explain your thinking process in detail.
    2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
    """
    elif "scensemble" in node_name.lower():
        return """
    Several answers have been generated to a same question. 
    Identify the concise answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.
    In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
    """
    elif "format" in node_name.lower():
        return """
    please extract a short and concise answer contains only Option words(A/B/C/D) from the following solution: {solution}.
    Make sure there are no additional comments or explanations in your response.
    """
    elif "problem" in node_name.lower():
        return "problem"
    elif "customcodegenerate" in node_name.lower():
        return "Fill CodeBlock"
    elif "test" in node_name.lower():
        return """
    Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: 
    ### problem
    {problem}

    ### Code Solution
    {solution}

    ### Execution Result
    {exec_pass}

    #### Failed Test Case
    {test_fail}

    Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.
    """
    elif "programmer" in node_name.lower():
        return """
        You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.
        """
    
def update_node_list(node_list, node_name, prompt):
    new_node_list = []
    node_found = False
    
    # 遍历原始节点列表
    for node in node_list:
        if node["name"] == node_name:
            # 如果找到了匹配的节点，更新其 prompt
            new_node_list.append({"name": node_name, "prompt": prompt})
            node_found = True
        else:
            new_node_list.append(node)
    return new_node_list
        

def add_node(node_list, node_name, prompt):
    # 如果没有找到该节点，则将其添加到列表中
    new_node_list = []
    for node in node_list:
        if node["name"] == node_name:
            return node_list    
    new_node_list = node_list + [{"name": node_name, "prompt": prompt}]
    return new_node_list

def delet_0_edge_nodes(nodes_list, edges_list):
    new_nodes_list = []
    
    # 遍历每个节点
    for node in nodes_list:
        # 判断该节点是否既不是输入也不是输出
        if node["name"] not in [edge["input"] for edge in edges_list] and node["name"] not in [edge["output"] for edge in edges_list]:
            print(f"Node {node['name']} has no edges and is removed!")
        else:
            # 如果节点有边，保留节点
            new_nodes_list.append(node)
    
    # 返回删除后的节点列表
    return new_nodes_list

def num_0_outdegree_nodes(nodes_list, edges_list):
    count = 0
    out_node_list = []
    for node in nodes_list:
        if node["name"] not in [edge["input"] for edge in edges_list]:
            count += 1
            out_node_list.append(node["name"])
    return count, out_node_list

def check_node_prompt(node_list):
    error_prompt_list = ["custom", "direct", "assemble"]
    for node in node_list:
        if node['prompt'].lower() in error_prompt_list:
            print(f"Error prompt in node: {node['name']}")
            return False
    return True

def save_jsonl(path, dataset):
    with open(path, "w") as f:
        for data in dataset:
            json.dump(data, f)
            f.write('\n')

def save_json(path, dataset):
    with open(path, "w") as f:
        json.dump(dataset, f, indent=4)
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="experiments", help="base directory of the experiments")
    parser.add_argument("--dataset", type=str, default="HumanEval", help="dataset name")
    # parser.add_argument("--final_save_jsonl_path", type=str, default="./experiments/MMLU/new_labels.jsonl", help="path to save the final dataset in jsonl format")
    # parser.add_argument("--final_save_json_path", type=str, default="./experiments/MMLU/new_labels.json", help="path to save the final dataset in json format")
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d_%H%M%S"), help='Timestamp for directory naming')
    args = parser.parse_args()
    return args

def count_labels(path):
    with open(path, "r") as f:
        dataset = [json.loads(line) for line in f]
        print(f"Total labels: {len(dataset)}")
        count_1 = 0
        count_0 = 0
        for data in dataset:    
            if data["label"]:
                count_1 += 1
            else:
                count_0 += 1
        print(f"Labels with score >= 0.5: {count_1}")    
        print(f"Labels with score < 0.5: {count_0}")
if __name__ == "__main__":
    args = parse_args()
    # base_dirs = [os.path.join(args.base_dir, args.dataset, 'workflows_latest')]
    base_dirs = ["experiments/HumanEval/deepseek_latest_labels"]
    dataset = args.dataset
    
    # unfinished_dir = os.path.join(args.base_dir, dataset, "unfinished_workflows_latest")
    # target_basedir = os.path.join(args.base_dir, dataset, "history", f"labels_{args.time}")
    # latest_target_basedir = os.path.join(args.base_dir, dataset, "labels_latest")
    # final_save_jsonl_path = os.path.join(args.base_dir, dataset, "labels.jsonl")
    # final_save_json_path = os.path.join(args.base_dir, dataset, "labels.json")
    unfinished_dir = os.path.join(args.base_dir, dataset, "deepseek", "unfinished_workflows_latest")
    target_basedir = os.path.join(args.base_dir, dataset, "history", f"labels_{args.time}")
    latest_target_basedir = os.path.join(args.base_dir, dataset, "deepseek", "labels_latest")
    final_save_jsonl_path = os.path.join(args.base_dir, dataset, "deepseek","labels.jsonl")
    final_save_json_path = os.path.join(args.base_dir, dataset, "deepseek","labels.json")

    os.makedirs(target_basedir)
    os.makedirs(unfinished_dir, exist_ok=True)
    dataset_len = get_dataset_len(dataset)
    dataset_list = get_dataset(dataset)
    task_list = [item["prompt"] for item in dataset_list]
        
    unfinished_subdir_list = []
    finished_subdir_list = []
    for base_dir in base_dirs:
        save_jsonl_path = f"./{base_dir}/labels.jsonl"
        save_json_path = f"./{base_dir}/labels.json"
        sub_dirs = []
        for sub_dir in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, sub_dir)):
                if sub_dir.startswith("round") and not os.path.exists(os.path.join(latest_target_basedir, sub_dir)):
                    sub_dirs.append(os.path.join(base_dir, sub_dir))
                # elif sub_dir.startswith("round") and os.path.exists(os.path.join(latest_target_basedir, sub_dir)):
                #     sub_dirs.append(os.path.join(latest_target_basedir, sub_dir))

        dataset = []
        
        count_0 = 0
        count_1 = 0
        for sub_dir in sub_dirs:
            # print(sub_dir)
            subdir_name = sub_dir.split("/")[-1]
            # if not subdir_name == "round_67":
            #     continue
            workflow_path = os.path.join(sub_dir, "workflow.json") if not os.path.exists(os.path.join(sub_dir, "mod_workflow.json")) else os.path.join(sub_dir, "mod_workflow.json")
            with open(workflow_path, "r") as f:
                workflow = json.load(f)
            nodes = workflow["nodes"]
            edges = workflow["edges"]
            nodes_list = []
            for node in nodes:
                node_name = node[0]
                prompt = get_nodes_prompt(node_name)
                nodes_list.append({"name": node_name, 
                                    "prompt": prompt})
                
            edges_list = []
            for edge in edges:
                input = edge[0]
                output = edge[1]
                instruction = edge[-1]["instruction"]
                if input not in [node["name"] for node in nodes_list]:
                    nodes_list = add_node(nodes_list, input, get_nodes_prompt(input))
                if output not in [node["name"] for node in nodes_list]:
                    nodes_list = add_node(nodes_list, output, get_nodes_prompt(output))
                if "custom" in output.lower() and "customcodegenerate" not in output.lower():
                    custom_node_name = output
                    custom_prompt = instruction
                    # if instruction == "direct":
                        # print(f"{sub_dir} have custom node with direct instruction: {custom_node_name}")
                    nodes_list = update_node_list(nodes_list, custom_node_name, custom_prompt)
                # elif ("custom" not in input.lower()) and  ("problem initialization" not in instruction) and ("direct" not in instruction):
                    # print(f"Custom node not found in edge: {edge}")
                edge_dict = {"input": input, 
                            "output": output, 
                            "instruction": instruction}
                edges_list.append(edge_dict)
                # add node if not in the node list
                
            
            nodes_list = delet_0_edge_nodes(nodes_list, edges_list)
            time.sleep(0.1)
            num_out_node, out_node_list = num_0_outdegree_nodes(nodes_list, edges_list)
            assert num_out_node == 1
            
            
            assert check_node_prompt(nodes_list)
            
                
            label_path = []
            for file in os.listdir(sub_dir):
                # file start with float number:
                if (file.startswith("0.") or file.startswith("1.")) and file.endswith(".json"):
                    avg_score = float(file.split("_")[0])
                    if avg_score != 0.0:
                        label_path.append(os.path.join(sub_dir, file))

            label_dict = {}
            for label_file in label_path:
                with open(label_file, "r") as f:
                    labels = json.load(f)
                for label in labels:
                    # task = label["input_text"]
                    # score = fix_score(label)
                    # prediction = label["prediction"]
                    # answer = label["answer"]
                    task = label["inputs"]
                    if task not in task_list:
                        print("task out of list!!!!")
                        continue
                    score = label["score"]
                    prediction = label["prediction"]
                    answer = label["expected_output"]
                    if task in label_dict:
                        label_dict[task].append({"score": score, "prediction": prediction, "answer": answer})
                    else:
                        label_dict[task] = [{"score": score, "prediction": prediction, "answer": answer}]
            
                
            # aggregate scores for each task
            
            assert len(label_dict) == dataset_len
            for task in label_dict:
                scores = [label["score"] for label in label_dict[task]]
                avg_score = sum(scores) / len(scores)
                label  = True if avg_score >= 0.5 else False
                if label:
                    count_1 += 1
                else:
                    count_0 += 1
                dataset.append({"task": task, "label": label, "nodes": nodes_list, "edges": edges_list})
            
            print(f"{sub_dir}_{len(label_dict)}Labels saved successfully!")

            finished_subdir_list.append(int(sub_dir.split("/")[-1].split("_")[-1]))
            shutil.copytree(sub_dir, os.path.join(target_basedir, subdir_name))
            if not os.path.exists(os.path.join(latest_target_basedir, subdir_name)):
                shutil.copytree(sub_dir, os.path.join(latest_target_basedir, subdir_name), dirs_exist_ok=False) 
        

        with open(save_jsonl_path, "w") as f:
            for data in dataset:
                json.dump(data, f)
                f.write('\n')
        with open(save_json_path, "w") as f:
            # json.dump(dataset, f)
            json.dump(dataset, f, indent=4)
        print(f"Total labels: {len(dataset)}")
        print(f"Labels with score >= 0.5: {count_1}")
        print(f"Labels with score < 0.5: {count_0}")
        # print(f"Dataset saved successfully to {save_jsonl_path} and {save_json_path}!")
        
        with open(final_save_jsonl_path, "w") as f:
            for data in dataset:
                json.dump(data, f)
                f.write('\n')
        with open(final_save_json_path, "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"{sub_dir} Dataset saved successfully to {final_save_json_path}!")
    print(f"Unfinished {len(unfinished_subdir_list)}subdirs: {unfinished_subdir_list}")
    print(f"Finished {len(finished_subdir_list)} subdirs: {finished_subdir_list}")
        


