import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score,f1_score
from torch_geometric.nn import GCNConv,GCN2Conv,GATConv,TransformerConv 
from convert_dataset_gnn import get_dataloader  
from predictor import Predictor,Predictor_concat    
import numpy as np
import random
from itertools import chain
import os

DIM_MAP = {"llama_3_1_8b":4096,"ST":384}    





def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, loader, optimizer, device):
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    i =0

    if isinstance(loader, list):
        num_batches = len(loader[0])+len(loader[1])
        loader = chain(*loader) 

    else:
        num_batches = len(loader)
    for batch in tqdm(loader,leave=False):
        model.train()
        batch = batch.to(device)
        optimizer.zero_grad()
        num_graphs = batch.batch[-1] + 1    
        batch.task_embedding = batch.task_embedding.reshape(num_graphs, -1)
        score = model(batch.x, batch.task_embedding, batch.edge_index, batch.batch)
        loss = F.binary_cross_entropy(score, batch.y.to(torch.float32))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = (score > 0.5).float()  
        correct_predictions += (preds == batch.y).sum().item()
        total_predictions += batch.y.size(0)
        batch_acc = accuracy_score(batch.y.cpu().numpy(),preds.cpu().numpy())

        i+=1
    avg_loss = total_loss / num_batches
    accuracy = correct_predictions / total_predictions
    return avg_loss, accuracy

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)   
            num_graphs = batch.batch[-1] + 1    
            batch.task_embedding = batch.task_embedding.reshape(num_graphs, -1)
            score = model(batch.x, batch.task_embedding, batch.edge_index, batch.batch)
            loss = F.binary_cross_entropy(score, batch.y.float())
            total_loss += loss.item()
            preds = (score > 0.5).float()
            y_true.extend(batch.y.cpu().numpy())    
            y_pred.extend(preds.cpu().numpy())
            correct_predictions += (preds == batch.y).sum().item()
            total_predictions += batch.y.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct_predictions / total_predictions
    model.train()
    return avg_loss, accuracy




def calculate_f1_score(model, loader, device):
    # Precision and Recall
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


def get_base_conv(base_conv):
    base_conv_dict = {'GCNConv': GCNConv, 'GCN2Conv': GCN2Conv,"GATConv":GATConv,"TransformerConv":TransformerConv}
    return base_conv_dict[base_conv]

def get_model(args):
    base_conv = get_base_conv(args.base_conv)
    if args.arch == 'concat':
        model = Predictor_concat(input_dim=args.input_dim, hidden_dim=args.hidden_dim, 
                      n_layers=args.n_layers,n_mlplayers=args.n_mlplayers, dropout=args.dropout,base_conv=base_conv)
    else:
        model = Predictor(input_dim=args.input_dim, hidden_dim=args.hidden_dim, 
                      n_layers=args.n_layers,n_mlplayers=args.n_mlplayers, dropout=args.dropout,base_conv=base_conv)
    return model    




def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    args.input_dim = DIM_MAP[args.llm]  
    print(f'Using input dim: {args.input_dim}')
    import os
    args.domain = os.path.basename(args.data_path)
    if args.domain=='Reason-GD':         
        branches = ['train_1','train_2','val']
        train_loader1,train_loader2,val_loader = get_dataloader(args,branches)
        train_loaders = [train_loader1,train_loader2] 
    else:
        branches = ['train','val']
        train_loader, val_loader = get_dataloader(args,branches)
        train_loaders = train_loader   

    print(args.base_conv)
    model = get_model(args).to(device)  
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0    
    
    best_model_dir = args.data_path + '/ckpt' + f'/{args.base_conv}_{args.arch}'
    import os 
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    best_model_path = os.path.join(best_model_dir, 'best_model.pth')
    for epoch in tqdm(range(1, args.epochs + 1),leave=False,desc='Epochs'):
        train_loss, train_acc = train(model, train_loaders, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model with Val Acc: {val_acc:.4f} at epoch {epoch}\r')
        
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN model on custom graph dataset')
    parser.add_argument('--data_path', type=str, default='./datasets_checkpoints/Coding-AF', help='Path to the root data directory')
    parser.add_argument('--llm', type=str, default="ST")
    parser.add_argument('--input_dim', type=int, default=384, help='Input feature dimension')
    parser.add_argument('--base_conv',type=str,default='GCNConv',help='Base convolution layer')
    parser.add_argument('--arch',type=str,default='concat',help='architecture of GNN')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducibility')
    parser.add_argument('--n_mlplayers', type=int, default=2, help='Number of MLP layers for task embedding'),
    args = parser.parse_args()
    main(args)
 