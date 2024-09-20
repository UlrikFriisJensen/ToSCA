#%% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential as pyg_Sequential, GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from torch.nn import Sequential, Conv1d

#%% Model

class SCVAE(torch.nn.Module):
    def __init__(self, ):
        super(SCVAE, self).__init__()
        self.graph_encoder_local = pyg_Sequential('x, edge_index, edge_attr', [
            (GATv2Conv(128, 128, heads=8, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.ELU(),
            (GATv2Conv(128*8, 128, heads=8, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.ELU(),
        ])
        
        self.graph_encoder_global = pyg_Sequential('x, edge_index, edge_attr', [
            (GATv2Conv(128, 128, heads=8, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.ELU(),
            (GATv2Conv(128*8, 128, heads=8, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.ELU(),
            (GATv2Conv(128*8, 128, heads=8, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.ELU(),
            (GATv2Conv(128*8, 128, heads=8, concat=True), 'x, edge_index, edge_attr -> x'),
            nn.ELU(),
        ])
        
        self.scattering_encoder = Sequential(
            Conv1d(2, 128, 7, padding=1),
            nn.ELU(),
            Conv1d(128, 128, 7, padding=1),
            nn.ELU(),
            nn.MaxPool1d(5, stride=3, padding=1),
            Conv1d(128, 256, 7, padding=1),
            nn.ELU(),
            Conv1d(256, 256, 7, padding=1),
            nn.ELU(),
        )
        
        self.combined_encoder = Sequential(
            nn.Linear(128*2 + 256*2, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
        )
        
        

    def encode(self, x, edge_index, edge_attr, batch):
        
        z_local = self.graph_encoder_local(x, edge_index, edge_attr)
        z_global = self.graph_encoder_global(z_local, edge_index, edge_attr)
        
        z_local = torch.cat([global_mean_pool(z_local, batch), global_max_pool(z_local, batch), global_add_pool(z_local, batch)], dim=1)
        
        z_graph = torch.cat((z_local, z_global), dim=1)
        
        return
    
    def reparameterize(self, mean, log_std):
        std = F.softplus(log_std) #torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def decode(self):
        return

    def forward(self, data):
        mu, logvar = self.encoder(data)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, self.classifier(z)
    