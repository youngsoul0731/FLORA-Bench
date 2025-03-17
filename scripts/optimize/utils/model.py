from scripts.utils.predictor import Predictor_ofa,Predictor,Predictor_concat 
from torch_geometric.nn import GCNConv, global_mean_pool,GCN2Conv,GATConv,TransformerConv
import argparse
import torch
import os
DIM_MAP = {"llama_3_1_8b":4096,"ST":384}
def get_base_conv(base_conv):
    base_conv_dict = {'GCNConv': GCNConv, 'GCN2Conv': GCN2Conv,"GATConv":GATConv,"TransformerConv":TransformerConv}
    return base_conv_dict[base_conv]

def parse_args(model_name):
    parser = argparse.ArgumentParser(description='Train GNN model on custom graph dataset')
    parser.add_argument('--base_conv',type=str,default='GCNConv',help='Base convolution layer')
    parser.add_argument('--arch',type=str,default='concat',help='architecture of GNN')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.8, help='Dropout rate')
    parser.add_argument('--n_mlplayers', type=int, default=2, help='Number of MLP layers for task embedding')
    parser.add_argument('--llm', type=str, default="ST", help='LLM model name')   
    parser.add_argument('--model_name', type=str, default=model_name, help='Model')
    args = parser.parse_args()
    return args

def get_model(model_name: str = None, ckp_base_path: str = None):  
    args = parse_args(model_name)  
    args.input_dim = DIM_MAP[args.llm]  
    base_conv = get_base_conv(args.base_conv)
    ckp_path = os.path.join(ckp_base_path + f'/{args.base_conv}_{args.arch}', f"best_model.pth")
    if args.arch == 'concat':
        model = Predictor_concat(input_dim=args.input_dim, hidden_dim=args.hidden_dim, 
                          n_layers=args.n_layers,n_mlplayers=args.n_mlplayers, dropout=args.dropout,base_conv=base_conv)
    
    else:
        model = Predictor(input_dim=args.input_dim, hidden_dim=args.hidden_dim, 
                          n_layers=args.n_layers,n_mlplayers=args.n_mlplayers, dropout=args.dropout,base_conv=base_conv)
    if ckp_path is not None:
        model.load_state_dict(torch.load(ckp_path))
    else:   
        # initial model
        model.reset_parameters()

    return model   