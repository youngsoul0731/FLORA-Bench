import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,global_mean_pool
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,GCN2Conv,GATConv,TransformerConv

class Predictor(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, n_layers=2,n_mlplayers=2, dropout=0.5,base_conv=GCNConv):
        super(Predictor, self).__init__()
        self.gnn_layers = nn.ModuleList()
        self.base_conv = base_conv  
        if self.base_conv in [GCNConv]:
            self.gnn_layers.append(base_conv(input_dim, hidden_dim,normalize=True))
            for _ in range(n_layers - 1):
                self.gnn_layers.append(base_conv(hidden_dim, hidden_dim,normalize=True))
        elif self.base_conv == GATConv:
            self.gnn_layers.append(base_conv(input_dim, hidden_dim//8,heads=8,concat=True))
            for i in range(n_layers - 1):
                self.gnn_layers.append(base_conv(hidden_dim, hidden_dim//8,heads=8,concat=True))   
        elif self.base_conv == GCN2Conv:
            self.gnn_layers.append(nn.Linear(input_dim, hidden_dim))
            for i in range(n_layers - 1):
                self.gnn_layers.append(base_conv(hidden_dim, alpha=0.1, shared_weights=True, normalize=True))
        elif self.base_conv == TransformerConv:
            self.gnn_layers.append(base_conv(input_dim, hidden_dim//8,heads=8,concat=True,beta=0.8))   
            for i in range(n_layers - 1):
                self.gnn_layers.append(base_conv(hidden_dim, hidden_dim//8,concat=True,heads=8,beta=0.8))
        self.dropout = dropout
        self.projector = nn.ModuleList()
        self.projector.append(nn.Linear(input_dim, hidden_dim))
        for i in range(n_mlplayers):
            self.projector.append(nn.Linear(hidden_dim, hidden_dim))
        self.final_mlp_layers = nn.ModuleList()
        
        self.final_mlp_layers.append(nn.Linear(hidden_dim*2, hidden_dim))
        for i in range(n_mlplayers-1):    
            self.final_mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.final_mlp_layers.append(nn.Linear(hidden_dim, 1))
    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()
        for proj in self.projector:
            proj.reset_parameters()
        for mlp in self.final_mlp_layers:
            mlp.reset_parameters()
    def forward(self, node_embedding, task_embedding, edge_index, batch):
        
        for i, gnn in enumerate(self.gnn_layers):
            if edge_index.dtype != torch.long:
                edge_index = edge_index.type(torch.long)
            if self.base_conv == GCN2Conv:
                if i == 0:
                    node_embedding = gnn(node_embedding)
                else:
                    node_embedding = gnn(node_embedding, node_embedding, edge_index)
            elif self.base_conv == GCNConv or GATConv:
                node_embedding = gnn(node_embedding, edge_index)
            elif self.base_conv == TransformerConv:
                node_embedding = gnn(node_embedding, edge_index)
            else:
                raise NotImplementedError
            if i != len(self.gnn_layers) - 1:
                node_embedding = F.relu(node_embedding)
                node_embedding = F.dropout(node_embedding, p=self.dropout, training=self.training)
        
        pooled_node_embedding = global_mean_pool(node_embedding, batch)
        pooled_node_embedding = F.normalize(pooled_node_embedding, p=2, dim=1)  
        # score =  nn.Linear(pooled_node_embedding.size(1),1).to('cuda')(pooled_node_embedding).to('cuda')
        if isinstance(task_embedding,list):
            task_embedding = np.array(task_embedding)
            task_embedding = torch.tensor(task_embedding).to('cuda')    

        for i, proj in enumerate(self.projector):
            task_embedding = proj(task_embedding)
            if i != len(self.projector) - 1:
                task_embedding = F.relu(task_embedding)
                task_embedding = F.dropout(task_embedding, p=self.dropout, training=self.training)
        task_embedding = F.normalize(task_embedding, p=2, dim=1)
        # embedding = torch.cat([pooled_node_embedding,task_embedding],dim=1)
        
        
        # for i, proj in enumerate(self.final_mlp_layers):
        #     embedding = proj(embedding)
        #     if i != len(self.final_mlp_layers) - 1:
        #         embedding = F.relu(embedding)
        #         embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        # score = torch.sigmoid(embedding).reshape(-1)    
        if task_embedding.dim() == 1:
            task_embedding = task_embedding.unsqueeze(0).expand(pooled_node_embedding.size(0), -1)
        score = torch.sigmoid( (pooled_node_embedding * task_embedding).sum(dim=1))
        # score = torch.sigmoid(score).reshape(-1)
        return score




class Predictor_concat(Predictor):
    
    def __init__(self, input_dim, hidden_dim, n_layers=2, n_mlplayers=2, dropout=0.5, base_conv=GCNConv):
        super(Predictor_concat, self).__init__(input_dim, hidden_dim, n_layers, n_mlplayers, dropout, base_conv)
    
    
    def forward(self, node_embedding, task_embedding, edge_index, batch):
        for i, gnn in enumerate(self.gnn_layers):
            if edge_index.dtype != torch.long:
                edge_index = edge_index.type(torch.long)
            if self.base_conv == GCN2Conv:
                if i == 0:
                    node_embedding = gnn(node_embedding)
                else:
                    node_embedding = gnn(node_embedding, node_embedding, edge_index)
            elif self.base_conv == GCNConv or GATConv:
                node_embedding = gnn(node_embedding, edge_index)
            elif self.base_conv == TransformerConv:
                node_embedding = gnn(node_embedding, edge_index)
            else:
                raise NotImplementedError
            if i != len(self.gnn_layers) - 1:
                node_embedding = F.relu(node_embedding)
                node_embedding = F.dropout(node_embedding, p=self.dropout, training=self.training)
        
        pooled_node_embedding = global_mean_pool(node_embedding, batch)
        pooled_node_embedding = F.normalize(pooled_node_embedding, p=2, dim=1)  
        if isinstance(task_embedding,list):
            task_embedding = np.array(task_embedding)
            task_embedding = torch.tensor(task_embedding).to('cuda')    

        for i, proj in enumerate(self.projector):
            task_embedding = proj(task_embedding)
            if i != len(self.projector) - 1:
                task_embedding = F.relu(task_embedding)
                task_embedding = F.dropout(task_embedding, p=self.dropout, training=self.training)
        task_embedding = F.normalize(task_embedding, p=2, dim=1)
        embedding = torch.cat([pooled_node_embedding,task_embedding],dim=1)
        
        
        for i, proj in enumerate(self.final_mlp_layers):
            embedding = proj(embedding)
            if i != len(self.final_mlp_layers) - 1:
                embedding = F.relu(embedding)
                embedding = F.dropout(embedding, p=self.dropout, training=self.training)
        score = torch.sigmoid(embedding).reshape(-1)    
        # if task_embedding.dim() == 1:
        #     task_embedding = task_embedding.unsqueeze(0).expand(pooled_node_embedding.size(0), -1)
        # score = torch.sigmoid( (pooled_node_embedding * task_embedding).sum(dim=1))
        # score = torch.sigmoid(score).reshape(-1)
        return score
    
    
    
    def get_penultimate_layer(self, node_embedding, mask,edge_index, batch):   
        for i, gnn in enumerate(self.gnn_layers):
            if edge_index.dtype != torch.long:
                edge_index = edge_index.type(torch.long)
            if self.base_conv == GCN2Conv:
                if i == 0:
                    node_embedding = gnn(node_embedding)
                else:
                    node_embedding = gnn(node_embedding, node_embedding, edge_index)
            elif self.base_conv == GCNConv or GATConv:
                node_embedding = gnn(node_embedding, edge_index)
            elif self.base_conv == TransformerConv:
                node_embedding = gnn(node_embedding, edge_index)
            else:
                raise NotImplementedError
            if i != len(self.gnn_layers) - 1:
                node_embedding = F.relu(node_embedding)
                node_embedding = F.dropout(node_embedding, p=self.dropout, training=self.training)
        
        pooled_node_embedding = global_mean_pool(node_embedding, batch)
        pooled_node_embedding = F.normalize(pooled_node_embedding, p=2, dim=1)  
        return pooled_node_embedding
        
    


class Predictor_ofa(Predictor):
    
    def __init__(self, input_dim, hidden_dim, n_layers=2, n_mlplayers=2, dropout=0.5, base_conv=GCNConv):
        super(Predictor_ofa, self).__init__(input_dim, hidden_dim, n_layers, n_mlplayers, dropout, base_conv)
        self.projector = nn.ModuleList()
        for i in range(n_mlplayers-1):
            self.projector.append(nn.Linear(hidden_dim, hidden_dim))
        self.projector.append(nn.Linear(hidden_dim, 1))
        
        
        
    def forward(self, node_embedding, mask,edge_index, batch):
        
        for i, gnn in enumerate(self.gnn_layers):
            if edge_index.dtype != torch.long:
                edge_index = edge_index.type(torch.long)
            if self.base_conv == GCN2Conv:
                if i == 0:
                    node_embedding = gnn(node_embedding)
                else:
                    node_embedding = gnn(node_embedding, node_embedding, edge_index)
            elif self.base_conv == GCNConv or GATConv:
                node_embedding = gnn(node_embedding, edge_index)
            elif self.base_conv == TransformerConv:
                node_embedding = gnn(node_embedding, edge_index)
            else:
                raise NotImplementedError
            if i != len(self.gnn_layers) - 1:
                node_embedding = F.relu(node_embedding)
                node_embedding = F.dropout(node_embedding, p=self.dropout, training=self.training)
                
        succeed_mask = []
        for i_mask in mask:
            succeed_mask = succeed_mask + i_mask
        succeed_mask = torch.tensor(succeed_mask).to('cuda').reshape(-1,1)
        # noi_prompt_embedding = node_embedding[noi_prompt_mask.squeeze() == 1]
        succeed_embedding = node_embedding[succeed_mask.squeeze() == 1]
        # fail_embedding = node_embedding[fail_mask.squeeze() == 1]
        
        for i, proj in enumerate(self.projector):
            succeed_embedding = proj(succeed_embedding)
            if i != len(self.projector) - 1:
                succeed_embedding = F.relu(succeed_embedding)
                succeed_embedding = F.dropout(succeed_embedding, p=self.dropout, training=self.training)
        success_prob = torch.sigmoid(succeed_embedding).reshape(-1) 
        return success_prob
        
