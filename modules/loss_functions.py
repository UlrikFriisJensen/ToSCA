#%% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential as pyg_Sequential, GATv2Conv, global_mean_pool, global_max_pool, global_add_pool, GATConv
from torch_geometric.nn.aggr import MLPAggregation, SetTransformerAggregation, PowerMeanAggregation
from torch.nn import Sequential, Conv1d
from torch.distributions import Independent, Normal
from torch.distributions.kl import kl_divergence

#%% Functions
class weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights, reduction='mean'):
        unreduced_loss = F.mse_loss(inputs,targets,reduction='none')
        weighted_loss = unreduced_loss * weights
        if reduction == 'mean':
            return torch.mean(weighted_loss)
        elif reduction == 'sum':
            return torch.sum(weighted_loss)
        else:
            return weighted_loss
        
class weighted_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights, reduction='mean'):
        unreduced_loss = F.cross_entropy(inputs,targets,reduction='none')
        weighted_loss = unreduced_loss * weights
        if reduction == 'mean':
            return torch.mean(weighted_loss)
        elif reduction == 'sum':
            return torch.sum(weighted_loss)
        else:
            return weighted_loss