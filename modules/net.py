#%% Imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Sequential as pyg_Sequential, GATv2Conv, global_mean_pool, global_max_pool, global_add_pool, GATConv
from torch_geometric.nn.aggr import MLPAggregation, SetTransformerAggregation, PowerMeanAggregation
from torch.nn import Sequential, Conv1d
from torch.distributions import Independent, Normal
from torch.distributions.kl import kl_divergence
from torch_geometric.utils import to_dense_batch

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
    def __init__(
        self, 
        latent_dim=128, 
        out_dim=50, 
        prior_factor=1,
        gnn_dim=64, 
        gnn_heads=1, 
        gnn_edge_dim=1, 
        scattering_channels=2, 
        scattering_dim=512, 
        scattering_kernel_size=1, 
        scattering_stride=1, 
        scattering_padding=0, 
        composition_dim=64,
        decoder_hidden_dim=2048, 
        position_output_dim=3, 
        atom_output_dim=119, 
        cell_output_dim=6
    ):
        super(SCVAE, self).__init__()
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        self.prior_factor = prior_factor
        self.gnn_dim = gnn_dim
        self.gnn_heads = gnn_heads
        self.gnn_edge_dim = gnn_edge_dim
        self.scattering_channels = scattering_channels
        self.scattering_dim = scattering_dim
        self.scattering_kernel_size = scattering_kernel_size
        self.scattering_stride = scattering_stride
        self.scattering_padding = scattering_padding
        self.composition_dim = composition_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.position_output_dim = position_output_dim
        self.atom_output_dim = atom_output_dim
        self.cell_output_dim = cell_output_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.aggr_list = ['sum'] #['mean', 'max', 'sum', 'std', 'var']
        
        self.graph_encoder_local = pyg_Sequential('x, edge_index, edge_attr', [
            (GATv2Conv(7, self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim), 'x, edge_index, edge_attr -> x'), # aggr=self.aggr_list
            nn.ELU(),
            (GATv2Conv(self.gnn_dim*self.gnn_heads*len(self.aggr_list), self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim), 'x, edge_index, edge_attr -> x'),
            nn.ELU(),
        ])
        
        # self.graph_encoder_global = pyg_Sequential('x, edge_index, edge_attr', [
        #     (GATv2Conv(self.gnn_dim*self.gnn_heads*len(self.aggr_list), self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim), 'x, edge_index, edge_attr -> x'),
        #     nn.ELU(),
        #     (GATv2Conv(self.gnn_dim*self.gnn_heads*len(self.aggr_list), self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim), 'x, edge_index, edge_attr -> x'),
        #     # nn.ELU(),
        #     # (GATv2Conv(self.gnn_dim*self.gnn_heads*len(self.aggr_list), self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim), 'x, edge_index, edge_attr -> x'),
        #     # nn.ELU(),
        #     # (GATv2Conv(self.gnn_dim*self.gnn_heads*len(self.aggr_list), self.gnn_dim, heads=self.gnn_heads, concat=True, edge_dim=self.gnn_edge_dim), 'x, edge_index, edge_attr -> x'),
        #     # nn.ELU(),
        # ])
        
        self.graph_encoder_mlp = Sequential(
            nn.Linear(self.gnn_dim*self.gnn_heads*len(self.aggr_list)*self.out_dim, self.gnn_dim*4),
            nn.ELU(),
            nn.Linear(self.gnn_dim*4, self.gnn_dim),
            nn.ELU(),
        )
        
        self.linear_encoder = Sequential(
            # nn.Linear(self.gnn_dim*self.gnn_heads*len(self.aggr_list)*3, self.latent_dim*16),
            # nn.Linear(self.gnn_dim*self.gnn_heads*len(self.aggr_list)*3*2, self.latent_dim*16),
            # nn.Linear(self.gnn_dim*self.gnn_heads*len(self.aggr_list)*3 + self.scattering_dim // 8, self.latent_dim*16),
            # nn.Linear(self.scattering_dim // 8, self.latent_dim*16),
            # nn.Linear(self.gnn_dim, self.latent_dim*16),
            
            # nn.Linear(self.gnn_dim + self.scattering_dim // 8, self.latent_dim*16),
            # nn.ELU(),
            # nn.Linear(self.latent_dim*16, self.latent_dim*8),
            # nn.ELU(),
            # nn.Linear(self.latent_dim*8, self.latent_dim*4),
            # nn.ELU(),
            # nn.Linear(self.latent_dim*4, self.latent_dim*2),
            
            # nn.Linear(self.gnn_dim + self.scattering_dim // 8, latent_dim*64),
            # nn.ELU(),
            # nn.Linear(latent_dim*64, latent_dim*32),
            # nn.ELU(),
            # nn.Linear(latent_dim*32, latent_dim*16),
            # nn.ELU(),
            # nn.Linear(latent_dim*16, self.latent_dim*2),

            nn.Linear(self.gnn_dim + self.scattering_dim // 8 + self.composition_dim // 2, latent_dim*64),
            nn.ELU(),
            nn.Linear(latent_dim*64, latent_dim*32),
            nn.ELU(),
            nn.Linear(latent_dim*32, latent_dim*16),
            nn.ELU(),
            nn.Linear(latent_dim*16, self.latent_dim*2),
        )
        
        self.scattering_encoder = Sequential(
            GatedConv1d(6000, self.scattering_dim, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            nn.ELU(),
            GatedConv1d(self.scattering_dim, self.scattering_dim // 4, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            nn.ELU(),
            GatedConv1d(self.scattering_dim // 4, self.scattering_dim // 8, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            
            # GatedConv1d(6000, self.scattering_dim, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            # nn.ELU(),
            # GatedConv1d(self.scattering_dim, self.scattering_dim // 2, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            # nn.ELU(),
            # GatedConv1d(self.scattering_dim // 2, self.scattering_dim // 4, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            # nn.ELU(),
            # GatedConv1d(self.scattering_dim // 4, self.scattering_dim // 8, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            # nn.ELU(),
            # GatedConv1d(self.scattering_dim // 8, self.latent_dim*2, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
        )

        self.composition_encoder = Sequential(
            nn.Linear(self.atom_output_dim, self.composition_dim),
            nn.ELU(),
            nn.Linear(self.composition_dim, self.composition_dim // 2),
            nn.ELU(),
        )
        
        self.prior_scattering_encoder = Sequential(
            GatedConv1d(6000, self.scattering_dim * self.prior_factor, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            nn.ELU(),
            GatedConv1d(self.scattering_dim * self.prior_factor, self.scattering_dim * self.prior_factor // 2, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            nn.ELU(),
            GatedConv1d(self.scattering_dim * self.prior_factor // 2, self.scattering_dim * self.prior_factor // 4, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            nn.ELU(),
            GatedConv1d(self.scattering_dim * self.prior_factor // 4, self.scattering_dim * self.prior_factor // 8, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
            nn.ELU(),
            # GatedConv1d(self.scattering_dim // 8, self.latent_dim*2, self.scattering_kernel_size, self.scattering_stride, self.scattering_padding),
        )

        self.prior_composition_encoder = Sequential(
            nn.Linear(self.atom_output_dim, self.composition_dim * self.prior_factor),
            nn.ELU(),
            nn.Linear(self.composition_dim * self.prior_factor, self.composition_dim * self.prior_factor // 2),
            nn.ELU(),
        )

        self.prior_linear_encoder = Sequential(
            nn.Linear(self.scattering_dim * self.prior_factor // 8 + self.composition_dim * self.prior_factor // 2, latent_dim*64),
            nn.ELU(),
            nn.Linear(latent_dim*64, latent_dim*32),
            nn.ELU(),
            nn.Linear(latent_dim*32, latent_dim*16),
            nn.ELU(),
            nn.Linear(latent_dim*16, self.latent_dim*2),
        )
        
        self.shared_decoder = Sequential(
            # nn.Linear(self.latent_dim, self.decoder_hidden_dim//8),
            nn.Linear(self.latent_dim // 2, self.decoder_hidden_dim//8),
            nn.ELU(),
            nn.Linear(self.decoder_hidden_dim//8, self.decoder_hidden_dim//4),
            nn.ELU(),
            nn.Linear(self.decoder_hidden_dim//4, self.decoder_hidden_dim//2),
            nn.ELU(),
            nn.Linear(self.decoder_hidden_dim //2, self.decoder_hidden_dim),
            nn.ELU(),
        )
        
        self.cell_parameter_decoder = Sequential(
            nn.Linear(self.latent_dim // 2, self.decoder_hidden_dim//8),
            nn.ELU(),
            nn.Linear(self.decoder_hidden_dim//8, self.decoder_hidden_dim//4),
            nn.ELU(),
            nn.Linear(self.decoder_hidden_dim//4, self.cell_output_dim),

            # nn.Linear(self.decoder_hidden_dim, self.cell_output_dim),
        )
        
        self.cell_position_decoder = Sequential(
            # nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            # nn.ELU(),
            nn.Linear(self.decoder_hidden_dim, self.out_dim*self.position_output_dim),
        )
        
        self.cell_atom_decoder = Sequential(
            # nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            # nn.ELU(),
            nn.Linear(self.decoder_hidden_dim, self.out_dim*self.atom_output_dim),
        )
        

    def encode(self, x, edge_index, edge_attr, batch, scattering, composition):
        
        z_local = self.graph_encoder_local(x, edge_index, edge_attr)
        # z_global = self.graph_encoder_global(z_local, edge_index, edge_attr)
        
        z_local, _ = to_dense_batch(z_local, batch, max_num_nodes=self.out_dim, batch_size=batch.amax()+1)
        z_local = z_local.contiguous().view(batch.amax()+1, -1)
        z_local = self.graph_encoder_mlp(z_local)
        
        # z_local = torch.cat([global_mean_pool(z_local, batch), global_max_pool(z_local, batch), global_add_pool(z_local, batch)], dim=1)
        # z_local = self.local_aggregator(z_local, batch, dim_size=self.latent_dim*2)
        
        # z_global = torch.cat([global_mean_pool(z_global, batch), global_max_pool(z_global, batch), global_add_pool(z_global, batch)], dim=1)
        # z_global = self.global_aggregator(z_global, batch, dim_size=self.latent_dim*2)
        
        # z_posterior = torch.cat((z_local, z_global), dim=1)
        
        z_scattering = self.scattering_encoder(scattering)
        z_scattering = z_scattering.squeeze(-1)

        if composition == None:
            z_posterior = torch.cat((z_local, z_scattering), dim=1)
        else:
            z_composition = self.composition_encoder(composition)
            z_posterior = torch.cat((z_local, z_scattering, z_composition), dim=1)
        # z_posterior = torch.cat((z_local, z_scattering), dim=1)
        # z_posterior = z_local
        # z_posterior = z_scattering
        
        z_posterior = self.linear_encoder(z_posterior)
        
        post_mean, post_log_std = z_posterior.chunk(2, dim=-1)
        
        return post_mean, post_log_std
    
    def prior(self, scattering, composition):
        z_scattering = self.prior_scattering_encoder(scattering)
        z_scattering = z_scattering.squeeze(-1)

        if composition == None:
            z_prior = z_scattering
        else:
            z_composition = self.prior_composition_encoder(composition)
            z_prior = torch.cat((z_scattering, z_composition), dim=1)

        z_prior = self.prior_linear_encoder(z_prior)

        prior_mean, prior_log_std = z_prior.chunk(2, dim=-1)
        
        # prior_mean = torch.zeros((scattering.size(dim=0), self.latent_dim))
        # prior_log_std = torch.zeros((scattering.size(dim=0), self.latent_dim))
        
        return prior_mean, prior_log_std
    
    def reparameterize(self, mean, log_std):
        std = F.softplus(log_std) #torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        return mean + eps * std
        
    def decode(self, z, composition):
        
        latent_split = self.latent_dim // 2
        
        z_shared = z.clone()
        z_shared = self.shared_decoder(z_shared[:, :latent_split]) # z_shared[:, :latent_split]
        
        cell_parameters = self.cell_parameter_decoder(z[:, latent_split:]) # z[:, latent_split:]
        cell_parameters = cell_parameters.view(-1, self.cell_output_dim)
        
        cell_positions = self.cell_position_decoder(z_shared)
        cell_positions = cell_positions.view(-1, self.out_dim, self.position_output_dim)
        
        cell_atoms = self.cell_atom_decoder(z_shared)
        cell_atoms = cell_atoms.view(-1, self.out_dim, self.atom_output_dim)
        if composition != None:
            cell_atoms = cell_atoms + composition.unsqueeze(1)
        
        return cell_parameters, cell_positions, cell_atoms

    def forward(self, x, edge_index, scattering, composition, edge_attr=None, batch=None):
        # Posterior encoder
        post_mean, post_log_std = self.encode(x, edge_index, edge_attr, batch, scattering, composition)
        
        # Prior encoder
        prior_mean, prior_log_std = self.prior(scattering, composition)
        
        # Ensure no zero variance
        offset = 1e-15
        post_log_std = F.softplus(post_log_std) + offset
        prior_log_std = F.softplus(prior_log_std) + offset
        
        # Reparameterization
        posterior_dist = Independent(Normal(post_mean, post_log_std), 1)
        prior_dist = Independent(Normal(prior_mean, prior_log_std), 1)
        
        # Calculate KL divergence
        kld = kl_divergence(posterior_dist, prior_dist) / len(post_mean)
        
        # Sample from distribution
        z_sample = posterior_dist.rsample()
        
        # Decoder        
        cell_parameters, cell_positions, cell_atoms = self.decode(z_sample, composition)
        
        return cell_parameters, cell_positions, cell_atoms, kld, post_mean, post_log_std, prior_mean, prior_log_std, z_sample
    
    def predict(self, scattering, composition):
        # Prior encoder
        prior_mean, prior_log_std = self.prior(scattering, composition)
        
        # Ensure no zero variance
        offset = 1e-15
        prior_log_std = F.softplus(prior_log_std) + offset
        
        # Reparameterization
        prior_dist = Independent(Normal(prior_mean, prior_log_std), 1)
        
        # Sample from distribution
        z_sample = prior_dist.rsample()
        
        # Decoder
        cell_parameters, cell_positions, cell_atoms = self.decode(z_sample)
        
        return cell_parameters, cell_positions, cell_atoms, prior_mean, prior_log_std, z_sample