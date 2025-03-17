

import ast
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime
from collections import defaultdict


    
def extract_variable_names(node):
    """
    递归提取表达式中的变量名，包括列表、字典和嵌套结构
    """
    variable_names = set()

    if isinstance(node, ast.Name):
        # 单个变量名
        variable_names.add(node.id)

    elif isinstance(node, ast.Subscript):
        # 下标访问，例如 initial_solution['response']
        if isinstance(node.value, ast.Name):
            variable_names.add(node.value.id)  # 提取下标前的变量名
        # 继续递归解析下标中的其他部分（例如 keys 或 indices）
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                variable_names.update(extract_variable_names(value))

    elif isinstance(node, ast.List):
        # 列表解析
        for element in node.elts:
            variable_names.update(extract_variable_names(element))

    elif isinstance(node, ast.Dict):
        # 字典解析
        for value in node.values:
            variable_names.update(extract_variable_names(value))

    elif isinstance(node, (ast.BinOp, ast.Call, ast.Attribute, ast.JoinedStr, ast.FormattedValue)):
        # 操作符、调用、属性访问、字符串拼接等，递归子节点
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    variable_names.update(extract_variable_names(item))
            elif isinstance(value, ast.AST):
                variable_names.update(extract_variable_names(value))
                
    
    return variable_names

def extract_instruction_prompt(value_node):
    """
    提取`instruction`关键字参数中的文字内容。
    支持常量字符串、字符串拼接、变量替换等情况。
    """
    if isinstance(value_node, ast.Str):  # 直接是字符串
        return [value_node.s]
    elif isinstance(value_node, ast.Constant):  # Python 3.8+ 的字符串常量
        if isinstance(value_node.value, str):
            return [value_node.value]
    elif isinstance(value_node, ast.BinOp) and isinstance(value_node.op, ast.Add):  # 字符串拼接
        left = extract_instruction_prompt(value_node.left)
        right = extract_instruction_prompt(value_node.right)
        return (left or []) + (right or [])
    elif isinstance(value_node, ast.Call):  # 函数调用
        if isinstance(value_node.func, ast.Attribute) and value_node.func.attr == "format":
            # 针对字符串格式化调用，提取模板字符串
            if isinstance(value_node.func.value, ast.Str):
                return [value_node.func.value.s]
            elif isinstance(value_node.func.value, ast.Constant) and isinstance(value_node.func.value.value, str):
                return [value_node.func.value.value]
    elif isinstance(value_node, ast.JoinedStr):  # f-string 拼接
        parts = []
        for value in value_node.values:
            if isinstance(value, ast.Str):
                parts.append(value.s)
        return parts
    return []


class CallGraphParser(ast.NodeVisitor):
    def __init__(self, path):
        self.graph = nx.MultiDiGraph()
        self.current_function = None
        self.node_counter = {}
        self.variable_sources = {}
        self.instance_mapping = {}
        self.path = path
        self.imports = {}  # To store import aliases
        super().__init__()


    def visit_FunctionDef(self, node):
        """处理函数定义"""
        if node.name == "__init__":
            self.current_function = "__init__"
            self.generic_visit(node) 
            return

        if node.name == "__call__":
            self.current_function = node.name
            # 找到problem参数
            for arg in node.args.args:
                if arg.arg == "problem":
                    self.graph.add_node("problem", type="root")
            self.generic_visit(node)  

    def visit_AsyncFunctionDef(self, node):
        """处理异步函数定义"""
        self.visit_FunctionDef(node)
        
    def visit_For(self, node):
        """Handle for loops with fixed range iterators."""
        loop_count = self._get_loop_count(node)
        if loop_count:
            for i in range(loop_count):
                # Optionally, modify the node or context to reflect the iteration
                self.generic_visit(node)  # Process the loop body
        else:
            # If loop count is not fixed, process normally
            self.generic_visit(node)
            
    def visit_AsyncFor(self, node):
        """Handle async for loops with fixed range iterators."""
        loop_count = self._get_loop_count(node)
        if loop_count:
            for i in range(loop_count):
                self.generic_visit(node)
        else:
            self.generic_visit(node)
            
    def _get_loop_count(self, node):
        """
        Determine the number of iterations for loops like `for _ in range(3):`
        Returns the count as an integer if determinable, else None.
        """
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            if node.iter.func.id == "range" and len(node.iter.args) == 1:
                arg = node.iter.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    return arg.value
        return None

    def visit_Assign(self, node):
        """
        解析赋值语句：记录变量与函数调用关系
        """ 
        
        # 左侧变量名
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            output_var = node.targets[0].id
        elif len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute):
            output_var = node.targets[0].attr
        else:
            self.generic_visit(node)
            return
        
        iteration = getattr(self, 'current_iteration', None)
        if isinstance(node.value, ast.Call):
            # 处理类实例化，如 self.custom = operator.Custom(self.llm)
            if self.current_function == "__init__":
                self._handle_instance_assignment(output_var, node.value)
                return

        # 右侧是否是调用
        if isinstance(node.value, ast.Await) and isinstance(node.value.value, ast.Call):
            self._handle_call(output_var, node.value.value, iteration=iteration)
        if isinstance(node.value, ast.Call):
            self._handle_call(output_var, node.value, iteration=iteration)
        # elif isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Add):
        #     self._handle_call(output_var, node.value.right)
        if isinstance(node.value, ast.JoinedStr):
            self._handle_joined(output_var, node.value)

        if isinstance(node.value, ast.List):
            self.graph.add_node(output_var, type="list", description="List initialized")
            self.variable_sources[output_var] = output_var
            for elt in node.value.elts:
                self._handle_list_element(output_var, elt)



    def visit_Import(self, node):
        """Handle import statements to map aliases."""
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node):
        """Handle from ... import ... statements to map aliases."""
        module = node.module
        for alias in node.names:
            full_name = f"{module}.{alias.name}" if module else alias.name
            self.imports[alias.asname or alias.name] = full_name
    
    def visit_Expr(self, node):
        """
        处理单独的表达式，例如列表的 append 操作。
        """
        if isinstance(node.value, ast.Await) and isinstance(node.value.value, ast.Call):
            self._handle_append(None, node.value.value)
        elif isinstance(node.value, ast.Call):
            self._handle_append(None, node.value)
    
    def _reconstruct_fstring(self, node):
        """
        重建 f-string 的字符串内容，保留变量占位符
        """
        parts = []
        for value in node.values:
            if isinstance(value, ast.Str):
                parts.append(value.s)
            elif isinstance(value, ast.FormattedValue):
                var_names = extract_variable_names(value.value)
                if var_names:
                    parts.append(f"{{{var_names.pop()}}}")
        return "".join(parts)
    
    def _handle_list_element(self, parent_var, elt):

        if isinstance(elt, ast.Name):
            # 列表元素是一个变量，例如 'problem'
            var_name = elt.id
            self.graph.add_node(var_name, type="variable")
            self.graph.add_edge(var_name, parent_var, relationship="contains", instrcution = 'variable')
            # 记录变量来源
            self.variable_sources[var_name] = var_name

        elif isinstance(elt, ast.JoinedStr):
            
            instruction = self._reconstruct_fstring(elt)
            if instruction == "":
                instruction = "list element"
            variables = extract_variable_names(elt)
            for var in variables:
                if var in self.variable_sources:
                    # 添加边从变量指向列表变量，设置 instruction
                    self.graph.add_edge(var, parent_var, relationship="contains", instruction=instruction)
                else:
                    # 如果变量来源未知，默认从 "problem"
                    self.graph.add_edge("problem", parent_var, relationship="contains", instruction=instruction)
        elif isinstance(elt, ast.Constant) and isinstance(elt.value, str):
            # 列表元素是一个常量字符串，例如 "Fixed string"
            instruction = elt.value
            self.graph.add_edge("constant", parent_var, relationship="contains", instruction=instruction)
        elif isinstance(elt, ast.Str):  # 兼容 Python < 3.8
            instruction = elt.s
            self.graph.add_edge("constant", parent_var, relationship="contains", instruction=instruction)
        else:
           self.graph.add_edge("other", parent_var, relationship="contains", instruction="")


    
    def _handle_append(self, output_var, call_node):
        if isinstance(call_node.func, ast.Attribute):
            callee = call_node.func.attr
            if isinstance(call_node.func.value, ast.Name):  # 确定调用对象
                object_name = call_node.func.value.id

                # 处理列表的 append 操作
                if callee == "append" and object_name in self.variable_sources:
                    list_node = self.variable_sources[object_name]
                    args = [arg.value.id for arg in call_node.args if isinstance(arg.value, ast.Name)]
                    for arg in args:
                        if arg in self.variable_sources:
                            self.graph.add_edge(self.variable_sources[arg], list_node, type="append", instruction = 'append')
                        else:
                            self.graph.add_edge("problem", list_node, input_var=arg, type="append", instruction = 'append')
                    return

    def _handle_joined(self, output_var, node):
        for value in node.values:
            combined_inputs = extract_variable_names(value)
            self.graph.add_node(output_var, type="joined")
            for input_var in combined_inputs:
                if input_var in self.variable_sources:
                    self.graph.add_edge(self.variable_sources[input_var], output_var, type="combine", instruction = "joined")
                else:
                    self.graph.add_edge("problem", output_var, input_var=input_var, type="combine", instruction = "joined")
            
            # 更新变量来源
            self.variable_sources[output_var] = output_var


    def _handle_instance_assignment(self, target_var, call_node):
        """
        处理对象实例化，记录实例与类的映射关系
        """
        if isinstance(call_node.func, ast.Attribute):
            # 处理如 operator.Custom
            class_name = f"{call_node.func.value.id}.{call_node.func.attr}"
        elif isinstance(call_node.func, ast.Name):
            # 处理直接调用的类名
            class_name = call_node.func.id
        else:
            return

        self.instance_mapping[target_var] = class_name

    def extract_callee(self, call_node):
        """
        提取调用节点的函数名
        """
        callee = None
        if isinstance(call_node.func, ast.Attribute):
            callee = call_node.func.attr
            if isinstance(call_node.func.value, ast.Name):
                instance_name = call_node.func.value.id
                if callee in self.instance_mapping:
                    # 用完整类名替换节点名
                    # callee = f"{self.instance_mapping[instance_name]}.{callee}"
                    callee = f"{self.instance_mapping[callee]}"
                else:
                    callee = f"{instance_name}.{callee}"

            callee = f"{callee}_{self._get_node_count(callee)}"

        elif isinstance(call_node.func, ast.Name):
            callee = call_node.func.id
        else:
            print(f"Unsupported call node: {ast.dump(call_node)}")
        return callee
    
    def _handle_call(self, output_var, call_node, iteration=None):
        """
        处理调用节点，提取调用的输入和函数名
        """
        # 提取被调用方法名
        callee = self.extract_callee(call_node)

        inputs = []
        instructions = []

        for arg in call_node.args:
            # ignore entry_point argument
            if arg == "entry_point":
                continue
            var_names = extract_variable_names(arg)
            if var_names:
                # combine "promblem" and "entry_point" to "problem_entry_point"
                inputs.extend(var_names)
        for arg in call_node.keywords:
            if arg.arg == "entry_point":
                continue
            inputs.extend(extract_variable_names(arg.value))
            if arg.arg == "instruction":
                instructions.extend(extract_instruction_prompt(arg.value))
            
                
                
        if len(instructions) == 0 or instructions[0] == "":
            if len(inputs) > 1:
                instructions.append("assemble")
            # if len(inputs) == 1 and inputs[0] not in self.variable_sources:
            #     instructions.append("direct")
        instruction = " ".join(instructions)
        # 添加调用节点
        
        node_name = callee
        if iteration is not None:
            node_name = f"{callee}_{iteration}"
        self.graph.add_node(node_name, type="call")
        
        for input_var in inputs:
            if input_var in self.variable_sources:
                if instruction == "":
                    instruction = "direct"
                self.graph.add_edge(
                    self.variable_sources[input_var],
                    node_name,
                    input_var=input_var,
                    output_var=output_var,
                    instruction=instruction
                )
            else:
                if instruction == "":
                    instruction = "problem initialization"
                self.graph.add_edge(
                    "problem",
                    node_name,
                    input_var=input_var,
                    output_var=output_var,
                    instruction=instruction
                )

        # Update the source of the output variable
        self.variable_sources[output_var] = node_name

    def _get_node_count(self, callee):
        """
        获取节点计数器，用于区分不同的 self.custom 调用
        """
        if callee not in self.node_counter:
            self.node_counter[callee] = 0
        self.node_counter[callee] += 1
        return self.node_counter[callee]

    def parse(self, filename):
        """解析文件并生成图"""
        with open(filename, "r", encoding="utf-8") as file:
            try:
                tree = ast.parse(file.read())
                self.visit(tree)
            except SyntaxError as e:
                print(f"Syntax error in {filename}: {e}")


    def extract_graph_data(self):
        """提取节点和边数据"""
        nodes = list(self.graph.nodes(data=True))
        edges = list(self.graph.edges(data=True))
        return nodes, edges
