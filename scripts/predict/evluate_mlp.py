import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,roc_auc_score, roc_curve
import os
import numpy as np
import random
from itertools import chain  
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from utils import JSONLReader
from prompt_embedding import get_encoder
from pathlib import Path


DIM_MAP = {"llama_3_1_8b":4096,"ST":384}    


class CustomDataset(Dataset):
    
    def __init__(self, jsonl_path): 
        data = JSONLReader.parse_file(jsonl_path)  
        memory_path = f"{Path(jsonl_path).parent}/memory_mlp.pkl"  
        memory = torch.load(memory_path) if os.path.exists(memory_path) else {} 
        encoder = get_encoder('ST', cache_dir="./model", batch_size=1)
        if not os.path.exists(memory_path):
            for item in data:
                workflow = str(item["nodes"]) + str(item["edge_index"])   
                if workflow not in memory.keys():
                    memory[workflow] = None  
                if item["task"] not in memory.keys():
                    memory[item["task"]] = None 
            attrs = list(memory.keys())
            attr_embedding = encoder.encode(attrs)
            for attr in attrs:
                memory[attr] = attr_embedding[attrs.index(attr)]   
            torch.save(memory, memory_path) 
            
        self.x = []
        self.y = []
        self.workflow_ids = []
        self.task_ids = []
        for item in data:
            workflow = str(item["nodes"]) + str(item["edge_index"]) 
            task = item["task"]
            try:
                workflow_feature = memory[workflow]
                task_feature = memory[task] 
            except:
                workflow_feature = encoder.encode([workflow])
                task_feature = encoder.encode([task])   
                memory[workflow] = workflow_feature
                memory[task] = task_feature 
            feature = torch.cat([workflow_feature,task_feature],dim=0)      
            self.x.append(feature.reshape(-1))            
            self.y.append(torch.tensor(item['label'], dtype=torch.long))    
            self.workflow_ids.append(item["workflow_id"])
            self.task_ids.append(item["task_id"]) 
        torch.save(memory, memory_path)
        

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.workflow_ids[idx], self.task_ids[idx]



class MLP_Predictor(nn.Module): 
    def __init__(self, input_dim, hidden_dim, n_layers=2, dropout=0.5):  
        super(MLP_Predictor, self).__init__()
        self.mlp_layers = nn.ModuleList()   
        self.mlp_layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_layers-2):
            self.mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.mlp_layers.append(nn.Linear(hidden_dim, 1))
        self.dropout = dropout

    def forward(self, x):
        for i,layer in enumerate(self.mlp_layers):
            x = layer(x)
            if i != len(self.mlp_layers)-1:
                x = F.relu(x)   
                x = F.dropout(x, p=self.dropout, training=self.training)    
                
        return torch.sigmoid(x).reshape(-1)

    
def get_dataloader(args,branches):
    base_dir = args.data_path
    loaders = []    
    for branch in branches:
        dataset_path = os.path.join(base_dir, f"{branch}.pt")   
        dataset = torch.load(dataset_path)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True if 'train' in branch else False)
        loaders.append(loader)
    return loaders
    

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, loader, optimizer, device,acc_lst=[]):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    if isinstance(loader, list):
        num_batches = len(loader[0])+len(loader[1])
        loader = chain(*loader) 

    else:
        num_batches = len(loader)
    model.train()    
    for x_batch, y_batch in tqdm(loader,leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        score = model(x_batch)
        loss = F.binary_cross_entropy(score, y_batch.to(torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = (score > 0.5).float()  
        correct_predictions += (preds == y_batch).sum().item()
        total_predictions += y_batch.size(0)
        # _,batch_acc = validate(model, val_loader, device)
        # plot_batch_acc(acc_lst)
    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy,acc_lst

@torch.no_grad()
def validate(args,model, loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred = [], []
    workflow_count_dict = {}    
    ground_workflow_dict = {}   
    predicted_workflow_dict = {} 
    with torch.no_grad():
        for batch_x,batch_y,batch_workflow_id,batch_task_id in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            score = model(batch_x)
            loss = F.binary_cross_entropy(score, batch_y.float())
            total_loss += loss.item()
            preds = (score > 0.5).float()
            batch_workflow_id = batch_workflow_id.cpu().tolist()    
            batch_task_id = batch_task_id.cpu().tolist()    
            for i, workflow_id in enumerate(batch_workflow_id):
                if workflow_id not in workflow_count_dict.keys():
                    workflow_count_dict[workflow_id] = 1
                else:
                    workflow_count_dict[workflow_id] += 1
                if workflow_id not in ground_workflow_dict.keys():
                    ground_workflow_dict[workflow_id] = 0
                if workflow_id not in predicted_workflow_dict.keys():
                    predicted_workflow_dict[workflow_id] = 0
                ground_workflow_dict[workflow_id] += batch_y[i].cpu().item()
                predicted_workflow_dict[workflow_id] += preds[i].cpu().item()   
            y_true.extend(batch_y.cpu().numpy())    
            y_pred.extend(preds.cpu().numpy())
            correct_predictions += (preds == batch_y).sum().item()
            total_predictions += batch_y.size(0)
    accuracy = correct_predictions / total_predictions
    ground_workflow_score_dict = {k:ground_workflow_dict[k]/workflow_count_dict[k] for k in ground_workflow_dict.keys()}    
    predicted_workflow_score_dict = {k:predicted_workflow_dict[k]/workflow_count_dict[k] for k in predicted_workflow_dict.keys()} 
    utility = calculate_utility(args,ground_workflow_score_dict,predicted_workflow_score_dict)
    return  accuracy, utility 


  
def precision_at_k(ground_workflows, predicted_workflows, k):
    # 根据分数对ground_workflows排序
    ground_sorted = sorted(ground_workflows.items(), key=lambda x: x[1], reverse=True)
    ground_top_k = set([workflow for workflow, score in ground_sorted[:k]])
    
    # 根据分数对predicted_workflows排序
    predicted_sorted = sorted(predicted_workflows.items(), key=lambda x: x[1], reverse=True)
    predicted_top_k = [workflow for workflow, score in predicted_sorted[:k]]
    
    # 计算Top K的Precision
    relevant_count = sum(1 for workflow in predicted_top_k if workflow in ground_top_k)
    precision = relevant_count / k
    
    return precision
    
def calculate_utility(args,ground_dict,predicted_dict):
    ground_workflows = sorted(ground_dict.items(), key=lambda x: x[1], reverse=True)
    predicted_workflows = sorted(predicted_dict.items(), key=lambda x: x[1], reverse=True)  
    ground_workflows = {k:v for k,v in ground_workflows}
    predicted_workflows = {k:v for k,v in predicted_workflows}   
    lst = []
    num_workflow = len(ground_dict)
    x = range(1, num_workflow+1)  
    perfect_overlap = [1.0]*len(x)
    perfect_auc = calculate_auc_precise(x, perfect_overlap)
    # for k in range(1, num_workflow+1):
    #     overlap_count = 0
    #     for workflow in list(predicted_workflows.keys())[:k]:
    #         if workflow in list(ground_workflows.keys())[:k]:
    #             overlap_count += 1
    #     lst.append(overlap_count / k)
    lst = [precision_at_k(ground_workflows,predicted_workflows,k) for k in x]
    utility = calculate_auc_precise(x, lst) / perfect_auc
    import matplotlib.pyplot as plt 
    save_path = os.path.join(args.data_path,'Precision@K_MLP.png')
    plt.ylabel("precision")
    plt.xlabel("k")
    plt.title("Precision@K")
    plt.xticks(np.arange(1, num_workflow+1, 10))
    plt.plot(x,lst,label=f'Predicted (AUC score: {utility:.2f})')
    plt.legend()

    plt.savefig(save_path)
    return utility
    
    
    
def calculate_auc_precise(x, y, num_points=100):
    interp_func = interp1d(x, y, kind='linear')
    x_interp = np.linspace(min(x), max(x), num_points)
    y_interp = interp_func(x_interp)
    auc = trapezoid(y_interp, x_interp)
    return auc











def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    args.input_dim = DIM_MAP[args.llm] * 2  
    print(f'Using input dim: {args.input_dim}')
    branches = ['test_mlp' ]
    test_loader = get_dataloader(args,branches)[0]
    # import pdb;pdb.set_trace()  
    model = MLP_Predictor(input_dim=args.input_dim, hidden_dim=args.hidden_dim, 
                      n_layers=args.n_layers,dropout=args.dropout).to(device)

    if args.cross_system:
        best_model_dir = args.cross_system + '/ckpt_mlp'    
    else:
        best_model_dir = args.data_path + '/ckpt_mlp'
    import os 

    os.makedirs(best_model_dir,exist_ok=True)
    best_model_path = os.path.join(best_model_dir, 'best_model.pth')

    model.load_state_dict(torch.load(best_model_path))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    print('Testing best model on test Set')
    test_acc,utility = validate(args,model, test_loader, device)
    print(f'Test Acc: {test_acc:.4f}')
    print(f"Utility: {utility:.4f}")
    print('-' * 50)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN model on custom graph dataset')
    parser.add_argument('--data_path', type=str, default='datasets_checkpoints/Coding-AF', help='Path to the root data directory')
    parser.add_argument('--cross_system', type=str, default='datasets_checkpoints/Coding-AF', help='model path when cross system')  
    parser.add_argument('--llm', type=str, default="ST")
    parser.add_argument('--input_dim', type=int, default=384, help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and evaluation')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args)
