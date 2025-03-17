

import ast
import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import argparse
from datetime import datetime
from collections import defaultdict
import importlib
import inspect


    
def extract_variable_names(node, excluded_names=None):
    """
    递归提取表达式中的变量名，包括列表、字典和嵌套结构
    """
    if excluded_names is None:
        excluded_names = set()
    variable_names = set()

    if isinstance(node, ast.Name):
        # 单个变量名
        if node.id not in excluded_names:
            variable_names.add(node.id)

    elif isinstance(node, ast.Subscript):
        # 下标访问，例如 initial_solution['response']
        if isinstance(node.value, ast.Name) and node.value.id not in excluded_names:
            variable_names.add(node.value.id)  # 提取下标前的变量名
        # 继续递归解析下标中的其他部分（例如 keys 或 indices）
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                variable_names.update(extract_variable_names(value, excluded_names))

    elif isinstance(node, ast.List):
        # 列表解析
        for element in node.elts:
            variable_names.update(extract_variable_names(element, excluded_names))

    elif isinstance(node, ast.Dict):
        # 字典解析
        for value in node.values:
            variable_names.update(extract_variable_names(value, excluded_names))

    elif isinstance(node, (ast.BinOp, ast.Call, ast.Attribute, ast.JoinedStr, ast.FormattedValue)):
        # 操作符、调用、属性访问、字符串拼接等，递归子节点
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    variable_names.update(extract_variable_names(item, excluded_names))
            elif isinstance(value, ast.AST):
                variable_names.update(extract_variable_names(value, excluded_names))
                
    
    return variable_names

def extract_instruction_prompt(value_node, current_module_globals=None):
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
        left = extract_instruction_prompt(value_node.left, current_module_globals)
        right = extract_instruction_prompt(value_node.right, current_module_globals)
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
    elif isinstance(value_node, ast.Attribute):  # 处理属性访问，如 prompt_custom.SOLUTION_PROMPT
        if current_module_globals is None:
            return []
        # 递归构建属性链
        attrs = []
        current = value_node
        while isinstance(current, ast.Attribute):
            attrs.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            attrs.insert(0, current.id)
            # 构建完整的属性路径
            full_attr = ".".join(attrs)
            # 尝试在 current_module_globals 中查找
            obj = current_module_globals.get(attrs[0], None)
            for attr in attrs[1:]:
                if obj is not None:
                    obj = getattr(obj, attr, None)
                else:
                    break
            if isinstance(obj, str):
                return [obj]
            else:
                return []
    return []


class CallGraphParser(ast.NodeVisitor):
    def __init__(self, path:str = None, obj:object = None):
        self.graph = nx.MultiDiGraph()
        self.current_function = None
        self.node_counter = {}
        self.variable_sources = {}
        self.instance_mapping = {}
        self.path = path
        self.imports = {}  # To store import aliases
        self.loaded_modules = {}
        self.obj = obj
        self.list_sources = defaultdict(list)
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
            
            self.variable_sources[output_var] = output_var
            self.list_sources[output_var] = []
            for elt in node.value.elts:
                var_names = extract_variable_names(elt, excluded_names=set())
                for var in var_names:
                    self.list_sources[output_var].append(self.variable_sources.get(var, var))
                
        if isinstance(node.value, ast.ListComp):
            self.variable_sources[output_var] = output_var
            
            iter_var = node.value.generators[0].iter.id
            for var in self.list_sources[iter_var]:
                self.list_sources[output_var].append(var)



    def visit_Import(self, node):
        """Handle import statements to map aliases."""
        for alias in node.names:
            name = alias.name
            asname = alias.asname or alias.name
            self.imports[asname] = name
            try:
                module = importlib.import_module(name)
                self.loaded_modules[asname] = module
                # print(f"Imported module: {name} as {asname}")
            except ImportError as e:
                print(f"Failed to import module {name}: {e}")

    def visit_ImportFrom(self, node):
        """Handle from ... import ... statements to map aliases."""
        module = node.module
        if module is None:
            return
        for alias in node.names:
            name = alias.name
            asname = alias.asname or alias.name
            full_name = f"{module}.{name}"
            self.imports[asname] = full_name
            try:
                loaded_module = importlib.import_module(module)
                obj = getattr(loaded_module, name, None)
                if obj is not None:
                    self.loaded_modules[asname] = obj
                    # print(f"Imported {name} from {module} as {asname}")
            except ImportError as e:
                print(f"Failed to import {name} from {module}: {e}")

    
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
                var_names = extract_variable_names(value.value, excluded_names=set(self.loaded_modules.keys()))
                if var_names:
                    parts.append(f"{{{var_names.pop()}}}")
        return "".join(parts)
    
    
    
    def _handle_append(self, output_var, call_node):
        if isinstance(call_node.func, ast.Attribute):
            callee = call_node.func.attr
            if callee == "append":
                list_name = call_node.func.value.id if isinstance(call_node.func.value, ast.Name) else None
                if list_name and list_name in self.variable_sources:
                    if len(call_node.args) != 1:
                        print(f"Unsupported append call: {ast.dump(call_node)}")
                        return
                    arg = call_node.args[0]
                    if isinstance(arg, ast.Subscript):
                        # 例如 solution['response']
                        var_names = extract_variable_names(arg, excluded_names=set(self.imports.keys()))
                        for var in var_names:
                            if var in self.variable_sources:
                                var  = self.variable_sources[var] 
                            self.list_sources[list_name].append(var)
                    elif isinstance(arg, ast.Name):
                        var = arg.id
                        if var not in self.imports:
                            if var in self.variable_sources:
                                var  = self.variable_sources[var] 
                            self.list_sources[list_name].append(var)
                    elif isinstance(arg, ast.Attribute):
                        var_names = extract_variable_names(arg, excluded_names=set(self.imports.keys()))
                        for var in var_names:
                            if var in self.variable_sources:
                                var  = self.variable_sources[var] 
                            self.list_sources[list_name].append(var)
                else:
                    pass

    def _handle_joined(self, output_var, node):
        for value in node.values:
            combined_inputs = extract_variable_names(value, excluded_names=set(self.loaded_modules.keys()))
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
            if isinstance(arg, ast.ListComp):
                var_names = extract_variable_names(arg.generators[0].iter, excluded_names=set(self.loaded_modules.keys()))
                inputs.extend(var_names)
                
                var_names = extract_variable_names(arg.elt, excluded_names=set(self.loaded_modules.keys()))
                inputs.extend(var_names)
                inputs = [v for v in inputs if v != 'range']
            
            if arg == "entry_point":
                continue
            var_names = extract_variable_names(arg, excluded_names=set(self.loaded_modules.keys()))
            if var_names:
                # combine "promblem" and "entry_point" to "problem_entry_point"
                inputs.extend(var_names)
        for arg in call_node.keywords:
            if arg.arg == "entry_point":
                continue
            inputs.extend(extract_variable_names(arg.value, excluded_names=set(self.loaded_modules.keys())))
            if arg.arg == "instruction":
                instructions.extend(extract_instruction_prompt(arg.value, self.loaded_modules))
            
                
                
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
        
        list_vars = [var for var in inputs if var in self.list_sources]
        non_list_vars = [var for var in inputs if var not in self.list_sources]

        for input_var in non_list_vars:
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
        for list_var in list_vars:
            sources = self.list_sources[list_var]
            for source_var in sources:
                self.graph.add_edge(
                    source_var,
                    node_name,
                    input_var=source_var,
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
        self.path = filename
        with open(filename, "r", encoding="utf-8") as file:
            try:
                tree = ast.parse(file.read())
                self.visit(tree)
            except SyntaxError as e:
                print(f"Syntax error in {filename}: {e}")


    def parse_obj(self, graph_obj):
        file = inspect.getfile(graph_obj)
        self.path = file
        self.parse(file)


    def extract_graph_data(self):
        """提取节点和边数据"""
        nodes = list(self.graph.nodes(data=True))
        edges = list(self.graph.edges(data=True))
        return nodes, edges
