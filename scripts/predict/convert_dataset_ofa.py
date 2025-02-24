import os 
import sys
import torch
import json
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from utils import JSONLReader
from prompt_embedding import get_encoder
from tqdm import tqdm
import time
import pickle
from sentence_transformers import SentenceTransformer




def get_dataloader(args,branches):
    loaders = []
    for branch in branches:
        branch2 = branch.split("_")[0]
        jsonl_path = args.data_path + f"/{branch2}.jsonl"
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


def construct_node_embedding(workflow: dict,node_attributes_memory: dict):
    features = []
    for node_id in workflow['nodes'].keys():
        prompt = workflow['nodes'][node_id]
        feature = node_attributes_memory[prompt]
        features.append(feature)
    features = torch.tensor(np.array(features))
    return features






class CustomGraphDataset(InMemoryDataset):
    def __init__(self, root, jsonl_path,branch, transform=None, pre_transform=None):
        self.jsonl_path = jsonl_path  
        self.branch = branch
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)

        self.data_list = torch.load(self.processed_paths[0],weights_only=False)

        
        
        
    @property
    def processed_file_names(self):
        return 'data.pt' 
    
    @staticmethod
    def data2vec(data,encoder):
        return encoder.encode(data)

    
    
    def process(self): 
        print(f"fist convert textual graph to PyG data from {self.jsonl_path}")
        memory_path = self.root.replace(f"/{self.branch}_ofa","") + "/memory_ofa.pkl"
        encoder = get_encoder('ST', cache_dir="./model", batch_size=1)
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

        noi_prompt = "Prompt Node. "+ "Graph classification on workflow performance."
        memory = {noi_prompt:None} if not os.path.exists(memory_path) else torch.load(memory_path)         
        classificaiton_prompt_1 = "Prompt Node. "+ "The workflow will succeed on the task: {task}" 
        classificaiton_prompt_2 = "Prompt Node. "+ "The workflow will fail on the task: {task}"
        tasks = []
        
        if not os.path.exists(memory_path): 
            for data in raw_data_lst:
                for node in data["nodes"].keys():
                    text = "Feature Node. "+ "System  prompt of the agent: " + data["nodes"][node]
                    if text not in memory.keys():
                        memory[text] = None 
                    task_text1 = classificaiton_prompt_1.format(task=data['task'])
                    task_text2 = classificaiton_prompt_2.format(task=data['task'])  
                    if task_text1 not in memory.keys():
                       memory[task_text1] = None
                    if task_text2 not in memory.keys():
                        memory[task_text2] = None
            node_attr = list(memory.keys())
            node_attr_embedding = encoder.encode(node_attr)
            for attr in node_attr:
                memory[attr] = node_attr_embedding[node_attr.index(attr)]   
            torch.save(memory, memory_path)
        print("embedding done")
 


        
        for workflow in tqdm(raw_data_lst,desc="Processing data"):
            num_feature_nodes = len(workflow['nodes'])  
            x = []
            for node in workflow['nodes'].keys():   
                text = "Feature Node. "+ "System  prompt of the agent: " + workflow["nodes"][node]
                try:
                    x.append(memory[text].reshape(-1))
                except:
                    x.append(encoder.encode([text]).reshape(-1))    
                    memory[text] = encoder.encode([text]).reshape(-1)
            x.append(memory[noi_prompt].reshape(-1))    
            task_text_1 = classificaiton_prompt_1.format(task=workflow['task'])
            task_text_2 = classificaiton_prompt_2.format(task=workflow['task'])
            try:
                current_label_feature_1 = memory[task_text_1]
                current_label_feature_2 = memory[task_text_2]   
            except:
                current_label_feature_1 = encoder.encode([task_text_1]).reshape(-1)
                current_label_feature_2 = encoder.encode([task_text_2]).reshape(-1)   
                memory[task_text_1] = current_label_feature_1
                memory[task_text_2] = current_label_feature_2
            x.append(memory[task_text_1].reshape(-1))   
            x.append(memory[task_text_2].reshape(-1))      
            x = torch.stack(x)
            init_edge_index = torch.tensor(workflow['edge_index']).t().contiguous()        
            noi_node_prompt_edge_index = torch.tensor([[i,num_feature_nodes] for i in range(num_feature_nodes)], dtype=torch.long)  
            noi_node_prompt_class_index_1 = torch.tensor([[num_feature_nodes,num_feature_nodes+1]], dtype=torch.long)   
            noi_node_prompt_class_index_2 = torch.tensor([[num_feature_nodes,num_feature_nodes+2]], dtype=torch.long)   
            noi_node_prompt_mask = [int(i==num_feature_nodes) for i in range(num_feature_nodes+3)]
            succeed_label_mask = [int(i==num_feature_nodes+1) for i in range(num_feature_nodes+3)]
            fail_label_mask = [int(i==num_feature_nodes+2) for i in range(num_feature_nodes+3)]  
            edge_index = torch.cat([init_edge_index,noi_node_prompt_edge_index.t(),noi_node_prompt_class_index_1.t().contiguous(),noi_node_prompt_class_index_2.t().contiguous()], dim=1) 
            masks = (noi_node_prompt_mask,succeed_label_mask,fail_label_mask)   
            data = Data(x=x, edge_index=edge_index, y=workflow['label'],mask = succeed_label_mask,workflow_id=workflow['workflow_id'],
                        task_id = workflow['task_id'])    
            data_list.append(data)
        raw_data_lst = None
        torch.save(memory, memory_path)    
        torch.save(data_list, self.processed_paths[0])
            
            
            
            
    def len(self):
        return len(self.data_list)


    



    
if __name__ == "__main__":
    
    for branch in ['train','val',"test"]:
        root = "datasets_checkpoints/Coding-AF"       
        jsonl_path = root + f"/{branch}.jsonl"  
        root = root + f"/{branch}_ofa"
        dataset = CustomGraphDataset(root=root, jsonl_path=jsonl_path,branch=branch)
