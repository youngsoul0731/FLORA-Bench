import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,roc_auc_score, roc_curve
from torch_geometric.nn import GCNConv,GCN2Conv,GATConv,TransformerConv 
from convert_dataset_ofa import get_dataloader  
from predictor import Predictor_ofa,Predictor   
import numpy as np
import random
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid

DIM_MAP = {"llama_3_1_8b":4096,"ST":384}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred = [], []
    workflow_count_dict = {}    
    ground_workflow_dict = {}   
    predicted_workflow_dict = {} 
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            score = model(batch.x, batch.mask,batch.edge_index, batch.batch)
            loss = F.binary_cross_entropy(score, batch.y.float())
            total_loss += loss.item()
            preds = (score > 0.5).float()
            batch.workflow_id = batch.workflow_id.cpu().tolist()    
            batch.task_id = batch.task_id.cpu().tolist()    
            for i, workflow_id in enumerate(batch.workflow_id):
                if workflow_id not in workflow_count_dict.keys():
                    workflow_count_dict[workflow_id] = 1
                else:
                    workflow_count_dict[workflow_id] += 1
                if workflow_id not in ground_workflow_dict.keys():
                    ground_workflow_dict[workflow_id] = 0
                if workflow_id not in predicted_workflow_dict.keys():
                    predicted_workflow_dict[workflow_id] = 0
                ground_workflow_dict[workflow_id] += batch.y[i].cpu().item()
                predicted_workflow_dict[workflow_id] += preds[i].cpu().item()   
            y_true.extend(batch.y.cpu().numpy())    
            y_pred.extend(preds.cpu().numpy())
            correct_predictions += (preds == batch.y).sum().item()
            total_predictions += batch.y.size(0)
    accuracy = correct_predictions / total_predictions
    ground_workflow_score_dict = {k:ground_workflow_dict[k]/workflow_count_dict[k] for k in ground_workflow_dict.keys()}    
    predicted_workflow_score_dict = {k:predicted_workflow_dict[k]/workflow_count_dict[k] for k in predicted_workflow_dict.keys()} 
    utility = calculate_utility(ground_workflow_score_dict,predicted_workflow_score_dict)
    return  accuracy, utility   




def calculate_f1_score(model, loader, device):
    # Precision and Recall
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            score = model(batch.x, batch.mask, batch.edge_index, batch.batch)
            preds = (score > 0.5).float()
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return f1_score(y_true, y_pred,average='binary')


def get_base_conv(base_conv):
    base_conv_dict = {'GCNConv': GCNConv, 'GCN2Conv': GCN2Conv,"GATConv":GATConv,"TransformerConv":TransformerConv}
    return base_conv_dict[base_conv]


def plot_batch_acc(acc_lst):
    import matplotlib.pyplot as plt
    plt.plot(acc_lst)
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.savefig('batch_acc.png')

def calculate_utility(ground_dict,predicted_dict):
    ground_workflows = sorted(ground_dict.items(), key=lambda x: x[1], reverse=True)
    predicted_workflows = sorted(predicted_dict.items(), key=lambda x: x[1], reverse=True)  
    ground_workflows = {k:v for k,v in ground_workflows}
    predicted_workflows = {k:v for k,v in predicted_workflows}   
    lst = []
    num_workflow = len(ground_dict)
    x = range(1, num_workflow+1)  
    perfect_overlap = [1.0]*len(x)
    perfect_auc = calculate_auc_precise(x, perfect_overlap)
    for k in range(1, num_workflow+1):
        overlap_count = 0
        for workflow in list(predicted_workflows.keys())[:k]:
            if workflow in list(ground_workflows.keys())[:k]:
                overlap_count += 1
        lst.append(overlap_count / k)
    
    utility = calculate_auc_precise(x, lst) / perfect_auc
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
    args.input_dim = DIM_MAP[args.llm]  
    print(f'Using input dim: {args.input_dim}')
    branches = ['test_ofa' ]
    test_loader = get_dataloader(args,branches)[0]

    base_conv = get_base_conv(args.base_conv)
    model = Predictor_ofa(input_dim=args.input_dim, hidden_dim=args.hidden_dim, 
                      n_layers=args.n_layers,n_mlplayers=args.n_mlplayers, dropout=args.dropout,base_conv=base_conv).to(device)

    if args.cross_system:
        best_model_dir = args.cross_system + '/ckpt_ofa' + f'/{args.base_conv}'
    else:
        best_model_dir = args.data_path + '/ckpt_ofa' + f'/{args.base_conv}'
    import os 
    assert os.path.exists(best_model_dir)

    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    best_model_path = os.path.join(best_model_dir, 'best_model.pth')
    
    model.load_state_dict(torch.load(best_model_path))
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    print('Testing best model on test Set')
    test_acc,utility = validate(model, test_loader, device)
    print(f'Test Acc: {test_acc:.4f}')
    print(f"Utility: {utility:.4f}")
    print('-' * 50)
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GNN model on custom graph dataset')
    parser.add_argument('--data_path', type=str, default='datasets_checkpoints/Coding-AF', help='Path to the root data directory')
    parser.add_argument('--llm', type=str, default="ST")
    parser.add_argument('--cross_system', type=str, default='datasets_checkpoints/Coding-AF', help='model path when cross system')
    parser.add_argument('--base_conv',type=str,default='GCNConv',help='Base convolution layer')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training and evaluation')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility')
    parser.add_argument('--n_mlplayers', type=int, default=2, help='Number of MLP layers for task embedding')
    args = parser.parse_args()
    main(args)
