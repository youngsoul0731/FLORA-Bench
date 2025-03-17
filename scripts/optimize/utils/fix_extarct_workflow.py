from typing import Dict, List, Tuple
from metagpt.configs.models_config import ModelsConfig
from scripts.utils.extract_MMLU_workflow import CallGraphParser
import time
import os



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

def delete_outdegree_0_nodes(nodes_list, edges_list, out_node_list):
    # hold the last 0 outdegree node(with edge in the later)
    last_0_outdegree_node = None
    new_nodes_list = []
    for node in out_node_list:
        if node in [edge["output"] for edge in edges_list]:
            last_0_outdegree_node = node
    for node in nodes_list:
        if node["name"] == last_0_outdegree_node:
            new_nodes_list.append(node)
        elif not node["name"] in out_node_list:
            new_nodes_list.append(node)

    new_edges_list = []
    for edge in edges_list:
        if edge["input"] or edge["output"] in new_nodes_list:
            new_edges_list.append(edge)
    return new_nodes_list, new_edges_list





def fix_test_extract_workflow(nodes: List, edges: List, graph_cls) -> Tuple[List, List]:

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
            if instruction == "direct":
                custom_prompt = input.lower()
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
    try:
        assert num_out_node == 1
    except:
        print(f"{num_out_node} nodes have 0 outdegree: {out_node_list} in class {graph_cls}")
        nodes_list, edges_list = delete_outdegree_0_nodes(nodes_list, edges_list, out_node_list)
        try:
            assert num_0_outdegree_nodes(nodes_list, edges_list)[0] == 1
        except:
            print(f"Error!!!!!!!!!!!!!!!!: {num_0_outdegree_nodes(nodes_list, edges_list)[0]} nodes have 0 outdegree in class {graph_cls}")
    try:
        assert check_node_prompt(nodes_list)
    except:
        print(f"Error!!!!!!!!!!!!!: Error prompt in class {graph_cls}")

    return nodes_list, edges_list