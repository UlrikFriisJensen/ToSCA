#%% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential as pyg_Sequential, GATv2Conv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.aggr import MLPAggregation, SetTransformerAggregation, PowerMeanAggregation
from torch.nn import Sequential, Conv1d

#%% Custom layers

class GatedConv1d(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size, stride, padding=0, dilation=1, activation=None):
        super(GatedConv1d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()

        self.h = Conv1d(input_channels, output_channels, kernel_size,
                           stride, padding, dilation)
        self.g = Conv1d(input_channels, output_channels, kernel_size,
                           stride, padding, dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation(self.h(x))
        g = self.sigmoid(self.g(x))

        return h * g

#%% Model

class SCVAE(nn.Module):
    def __init__(self, latent_dim=128, out_dim=50, gnn_dim=64, gnn_heads=1, gnn_edge_dim=1, scattering_channels=2, scattering_dim=256, scattering_kernel_size=7, scattering_stride=3, scattering_padding=1, decoder_hidden_dim=256):
        super(SCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.gnn_dim = gnn_dim
        self.gnn_heads = gnn_heads
        self.gnn_edge_dim = gnn_edge_dim
        self.scattering_channels = scattering_channels
        self.scattering_dim = scattering_dim
        self.scattering_kernel_size = scattering_kernel_size
        self.scattering_stride = scattering_stride
        self.scattering_padding = scattering_padding
        self.decoder_hidden_dim = decoder_hidden_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.aggr_list = ['mean', 'max', 'sum', 'std', 'var']
        
        self.graph_encoder_local = pyg_Sequential('x, edge_index, edge_attr', [
            (GATv2Conv(7, self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim, aggr=self.aggr_list), 'x, edge_index, edge_attr -> x'),
            nn.ELU(),
            (GATv2Conv(self.gnn_dim*self.gnn_heads*len(self.aggr_list), self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim, aggr=self.aggr_list), 'x, edge_index, edge_attr -> x'),
            nn.ELU(),
        ])
        
        # self.graph_encoder_global = pyg_Sequential('x, edge_index, edge_attr', [
        #     (GATv2Conv(self.gnn_dim*self.gnn_heads, self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim), 'x, edge_index, edge_attr -> x'),
        #     nn.ELU(),
        #     (GATv2Conv(self.gnn_dim*self.gnn_heads, self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim), 'x, edge_index, edge_attr -> x'),
        #     nn.ELU(),
        #     (GATv2Conv(self.gnn_dim*self.gnn_heads, self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim), 'x, edge_index, edge_attr -> x'),
        #     nn.ELU(),
        #     (GATv2Conv(self.gnn_dim*self.gnn_heads, self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim), 'x, edge_index, edge_attr -> x'),
        #     nn.ELU(),
        # ])
        
        self.linear_graph_encoder = Sequential(
            nn.Linear(self.latent_dim*self.gnn_heads*len(self.aggr_list)*3, self.latent_dim*16),
            nn.ELU(),
            nn.Linear(self.latent_dim*16, self.latent_dim*8),
            nn.ELU(),
            nn.Linear(self.latent_dim*8, self.latent_dim*4),
            nn.ELU(),
            nn.Linear(self.latent_dim*4, self.latent_dim*2),
        )
        
        self.scattering_encoder = Sequential(
            GatedConv1d(2, )
        )
        
        # self.local_aggregator = MLPAggregation(
        #     in_channels = self.gnn_dim*self.gnn_heads, 
        #     out_channels = self.latent_dim*2, 
        #     max_num_elements = self.out_dim,
        #     num_layers = 1
        #     ).to(self.device)
        
        # self.local_aggregator = PowerMeanAggregation(
        #     p=1.0,
        #     learn=True,
        #     channels=1,
        # )
        
        # self.global_aggregator = MLPAggregation(
        #     in_channels = self.gnn_dim*self.gnn_heads, 
        #     out_channels = self.latent_dim*2, 
        #     max_num_elements = self.out_dim,
        #     num_layers = 1
        #     ).to(self.device)
        
        # self.global_aggregator = PowerMeanAggregation(
        #     p=1.0,
        #     learn=True,
        #     channels=1,
        # )
        
        self.shared_decoder = Sequential(
            nn.Linear(self.latent_dim, self.decoder_hidden_dim),
            nn.ELU(),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.ELU(),
        )
        
        self.cell_parameter_decoder = Sequential(
            nn.Linear(self.decoder_hidden_dim, 6),
        )
        
        self.cell_position_decoder = Sequential(
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.ELU(),
            nn.Linear(self.decoder_hidden_dim, self.out_dim*3),
        )
        
        self.cell_atom_decoder = Sequential(
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.ELU(),
            nn.Linear(self.decoder_hidden_dim, self.out_dim*118),
        )
        

    def encode(self, x, edge_index, edge_attr, batch):
        
        z_local = self.graph_encoder_local(x, edge_index, edge_attr)
        z_global = self.graph_encoder_global(z_local, edge_index, edge_attr)
        
        z_local = torch.cat([global_mean_pool(z_local, batch), global_max_pool(z_local, batch), global_add_pool(z_local, batch)], dim=1)
        # z_local = self.local_aggregator(z_local, batch, dim_size=self.latent_dim*2)
        
        z_global = torch.cat([global_mean_pool(z_global, batch), global_max_pool(z_global, batch), global_add_pool(z_global, batch)], dim=1)
        # z_global = self.global_aggregator(z_global, batch, dim_size=self.latent_dim*2)
        
        z_graph = torch.cat((z_local, z_global), dim=1)
        
        z_graph = self.linear_graph_encoder(z_graph)
        
        mean, log_std = z_graph.chunk(2, dim=-1)
        
        return mean, log_std
    
    # def prior(self, scattering):
    #     z_scattering = self.scattering_encoder(scattering)
    
    def reparameterize(self, mean, log_std):
        std = F.softplus(log_std) #torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def decode(self, z):
        
        z = self.shared_decoder(z)
        
        cell_parameters = self.cell_parameter_decoder(z)
        cell_parameters = cell_parameters.view(-1, 6)
        
        cell_positions = self.cell_position_decoder(z)
        cell_positions = cell_positions.view(-1, self.out_dim, 3)
        
        cell_atoms = self.cell_atom_decoder(z)
        cell_atoms = cell_atoms.view(-1, self.out_dim, 118)
        
        return cell_parameters, cell_positions, cell_atoms

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        mean, log_std = self.encode(x, edge_index, edge_attr, batch)
        z = self.reparameterize(mean, log_std)
        cell_parameters, cell_positions, cell_atoms = self.decode(z)
        return cell_parameters, cell_positions, cell_atoms, mean, log_std
    