import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,roc_auc_score, roc_curve
import os
import numpy as np
import random
from itertools import chain
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
    for x_batch, y_batch,_, _ in tqdm(loader,leave=False):
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
    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy,acc_lst

@torch.no_grad()
def validate(model, loader, device):

    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_batch,y_batch,_,_ in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            score = model(x_batch)
            loss = F.binary_cross_entropy(score, y_batch.float())
            total_loss += loss.item()
            preds = (score > 0.5).float()
            y_true.extend(y_batch.cpu().numpy())    
            y_pred.extend(preds.cpu().numpy())
            correct_predictions += (preds == y_batch).sum().item()
            total_predictions += y_batch.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_predictions
    f1 = f1_score(y_true, y_pred,average='binary')
    model.train()
    return avg_loss, accuracy,f1




def calculate_f1_score(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            score = model(batch.x, batch.task_embedding, batch.edge_index, batch.batch)
            preds = (score > 0.5).float()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return f1_score(y_true, y_pred,average='binary')








def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    args.input_dim = DIM_MAP[args.llm] * 2  
    print(f'Using input dim: {args.input_dim}')
    import os
    args.domain = os.path.basename(args.data_path)
    
    base_dir = args.data_path
    for branch in ['train','val','test']:
        josnl_path = os.path.join(base_dir, f"{branch}.jsonl")
        branch = branch + "_mlp"
        dataset = CustomDataset(josnl_path)
        torch.save(dataset, os.path.join(base_dir, f"{branch}.pt"))
      
    if args.domain == 'Reason-GD':       
        branches = ['train_1_mlp','train_2_mlp','val_mlp']
        train_loader1,train_loader2,val_loader = get_dataloader(args,branches)
        train_loaders = [train_loader1,train_loader2] 
    else:
        branches = ['train_mlp','val_mlp']
        train_loader, val_loader = get_dataloader(args,branches)
        train_loaders = train_loader   
        
        
    
    model = MLP_Predictor(input_dim=args.input_dim, hidden_dim=args.hidden_dim, 
                      n_layers=args.n_layers,dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0    
    
    best_model_dir = args.data_path + '/ckpt_mlp'
    import os 

    os.makedirs(best_model_dir,exist_ok=True)
    best_model_path = os.path.join(best_model_dir, 'best_model.pth')
    acc_lst = []
    for epoch in tqdm(range(1, args.epochs + 1),leave=False,desc='Epochs'):
        train_loss, train_acc,acc_lst = train(model, train_loaders, optimizer, device,acc_lst)
        val_loss, val_acc, f1 = validate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with Val Acc: {val_acc:.4f} at epoch {epoch}\r')
        
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN model on custom graph dataset')
    parser.add_argument('--data_path', type=str, default='datasets_checkpoints/Coding-AF', help='Path to the root data directory') 
    parser.add_argument('--llm', type=str, default="ST")
    parser.add_argument('--input_dim', type=int, default=384, help='Input feature dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility')
    parser.add_argument('--n_mlplayers', type=int, default=2, help='Number of MLP layers for task embedding')
    args = parser.parse_args()
    main(args)
