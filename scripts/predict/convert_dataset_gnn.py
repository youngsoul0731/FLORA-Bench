
import os 
import sys
import torch
import json
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from sklearn.metrics import f1_score    

from utils import JSONLReader
from prompt_embedding import get_encoder
from tqdm import tqdm
import time
import pickle




def get_dataloader(args,branches):
    loaders = []
    for branch in branches:
        jsonl_path = args.data_path + f"/{branch}.jsonl"
        dataset = CustomGraphDataset(root=f"{args.data_path}/{branch}", jsonl_path=jsonl_path,branch=branch)
        print(f"length of {branch} dataset: {len(dataset)}")    
        loader = DataLoader(dataset.data_list, batch_size=args.batch_size, shuffle=True if 'train' in branch else False)
        loaders.append(loader)
    return loaders 







def process_pyg_dataset(dataset):
    for data in dataset:
        data.edge_index = data.edge_index.type(torch.long)



 


def  get_numerical_node_id(nodes: dict, node_id: str):
    return list(nodes.keys()).index(node_id)


def get_numerical_edge_index(nodes,str_edge_index):
    edge_index = []
    for edge in str_edge_index:
        # import pdb;pdb.set_trace()  
        edge_index.append([get_numerical_node_id(nodes,edge[0]),get_numerical_node_id(nodes,edge[1])])
    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index


def construct_node_embedding(workflow: dict,node_attributes_memory: dict,encoder):
    features = []
    for node_id in workflow['nodes'].keys():
        prompt = workflow['nodes'][node_id]
        try:
            feature = node_attributes_memory[prompt]
        except:
            feature = encoder.encode(prompt)
            node_attributes_memory[prompt] = feature
        features.append(feature.reshape(-1))
    features = torch.stack(features)
    return features, node_attributes_memory






class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, jsonl_path,branch, transform=None, pre_transform=None):
        self.jsonl_path = jsonl_path  
        self.branch = branch    
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)

        self.data_list = torch.load(self.processed_paths[0],weights_only=False)
        
    @property
    def processed_file_names(self):
        return 'data.pt' 
    
    
    
    def process(self):
        print(f"fist convert textual graph to PyG data from {self.jsonl_path}")
        encoder = get_encoder('ST', cache_dir="./model", batch_size=1)
        memory_path = self.root.replace(f"/{self.branch}","") + "/memory.pkl"
        data_list = []
        if not isinstance(self.jsonl_path, list):
            self.jsonl_path = [self.jsonl_path]
        raw_data_lst = []

        for jsonl_path in self.jsonl_path:
            if jsonl_path.endswith('.json'):
                with open(jsonl_path, 'r') as f:
                    raw_data_lst += json.load(f)
            elif jsonl_path.endswith('.jsonl'):
                raw_data_lst += JSONLReader.parse_file(jsonl_path) 
        memory = torch.load(memory_path) if os.path.exists(memory_path) else {}
        if not os.path.exists(memory_path):
            for data in tqdm(raw_data_lst,desc="Getting prompt embedding"):
                for node in data["nodes"].keys():
                    if data["nodes"][node] not in memory.keys():   
                        memory[data["nodes"][node]] = None
                if data["task"] not in memory.keys():
                    memory[data["task"]] = None 
            node_attr = list(memory.keys())
            node_attr_embedding = encoder.encode(node_attr)
            for attr in node_attr:
                memory[attr] = node_attr_embedding[node_attr.index(attr)]   
            torch.save(memory, memory_path)

        print("embedding done")
        
        



        
        for workflow in tqdm(raw_data_lst,desc="Processing data"):
            edge_index = torch.tensor(workflow['edge_index']).t().contiguous()  
            x = []
            
            for node_id in workflow['nodes'].keys():
                prompt = workflow['nodes'][node_id]
                try:
                    feature = memory[prompt]
                except:
                    feature = encoder.encode([prompt])
                    memory[prompt] = feature
                
                x.append(feature.reshape(-1))
            x = torch.stack(x)
            # x,memory = construct_node_embedding(workflow,memory,encoder) 
            num_nodes = len(workflow['nodes'])
            
            try: 
                task_embedding = memory[workflow['task']]
            except:
                task_embedding = encoder.encode([workflow['task']]).reshape(-1)
                memory[workflow['task']] = task_embedding 
                        
            y = torch.tensor(workflow['label'], dtype=torch.long)
            # additional_edge_index = torch.tensor([[num_nodes, i] for i in range(num_nodes)], dtype=torch.long)
            # edge_index = torch.cat([edge_index, additional_edge_index.t()], dim=1) 
            # x = torch.cat([x, torch.tensor(task_embedding).reshape(1,-1)], dim=0)
            data = Data(x=x, edge_index=edge_index, y=y,task_embedding = task_embedding, workflow_id=workflow['workflow_id'],
                        task_id = workflow['task_id'])  
            
            data.edge_index = data.edge_index.type(torch.long)
            data_list.append(data)
        raw_data_lst = None 
        torch.save(memory, memory_path)
        # data, slices = self.collate(data_list)
        # data.edge_index = data.edge_index.type(torch.long)

        torch.save(data_list, self.processed_paths[0])


    def len(self):
        return len(self.data_list)


    



    
if __name__ == "__main__":
    

    #     root += f"/{branch}"   
    for branch in ['train','val','test']:
        root = "datasets_checkpoints/Coding-AF"  
        jsonl_path = root + f"/{branch}.jsonl"  
        root = root + f"/{branch}"
        dataset = CustomGraphDataset(root=root, jsonl_path=jsonl_path,branch=branch)
