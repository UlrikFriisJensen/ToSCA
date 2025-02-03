#%% Imports
import argparse
import re
import json
import warnings
import numpy as np
from modules.CHILI import CHILI
from modules.net import SCVAE
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data import TensorDataset
import datetime
import pathlib
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ase import Atoms
from ase.io import write
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from modules.loss_functions import weighted_MSELoss, weighted_CrossEntropyLoss
from copy import deepcopy

#%% Suppress warnings
warnings.filterwarnings("ignore")

#%% Functions

def create_cif(cell_params, cell_positions, cell_atoms, filename, prediction=True, composition=None, simplified_atom_identities=False):
    """
    Create a CIF file from the cell parameters, positions and atoms
    """
    if prediction:
        # Find argmax of atoms
        cell_atoms = np.argmax(cell_atoms, axis=1)

    # Remove atoms with atom number 0
    cell_positions = cell_positions[cell_atoms != 0]
    cell_atoms = cell_atoms[cell_atoms != 0]
    
    # Remove atoms not in the unit cell
    cell_atoms = cell_atoms[(cell_positions[:,0] < 0.95) & (cell_positions[:,1] < 0.95) & (cell_positions[:,2] < 0.95)]
    cell_positions = cell_positions[(cell_positions[:,0] < 0.95) & (cell_positions[:,1] < 0.95) & (cell_positions[:,2] < 0.95)]
    
    
    if simplified_atom_identities:
        cell_atoms = np.where(cell_atoms == 1, 8, cell_atoms)
        cell_atoms = np.where(cell_atoms == 2, 26, cell_atoms)
    
    # Create Atoms object
    atoms = Atoms(cell_atoms, scaled_positions=cell_positions, cell=cell_params)

    if not composition:
        composition = str(atoms.symbols)

    # Write CIF
    write(filename + f'_{composition}.cif', images=atoms, format='cif')

    if not prediction:
        return composition
    return None

#%% Main

if __name__ == "__main__":
    #%% Parse arguments
    parser = argparse.ArgumentParser(description='Test the SCVAE model')
    parser.add_argument('--setup_json', type=str, help='Path to the setup json file')
    args = parser.parse_args()
    
    # Load setup json
    with open(args.setup_json, 'r') as f:
        setup_json = json.load(f)
        
    # Make predictions folder
    predictions_folder = f'{setup_json["model_root"]}{setup_json["experiment_name"]}/interpolation_predictions'
    pathlib.Path(predictions_folder).mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create latent samples for interpolation between two points
    n_latent_samples = 10
    latent_point_1 = (-0.5,1.5,-0.9)
    latent_point_2 = (2.5,1.5,-0.9)
    # Latent samples are n_latent_samples evenly spaced points between the two points
    interp_latent = torch.tensor([np.linspace(latent_point_1[i], latent_point_2[i], n_latent_samples) for i in range(3)]).T.float()
    interp_index = torch.tensor([i for i in range(n_latent_samples)])
    
    # Create dataset
    interp_dataset = TensorDataset(interp_latent, interp_index)
    
    # Dataloader
    interp_loader = DataLoader(interp_dataset, batch_size=10, shuffle=False)
    
    # Load model
    model = SCVAE(
        latent_dim=setup_json['model']['latent_dim'],
        out_dim=setup_json['model']['out_dim'],
        prior_factor=setup_json['model']['prior_factor'],
        gnn_dim=setup_json['model']['gnn_dim'],
        gnn_heads=setup_json['model']['gnn_heads'],
        gnn_edge_dim=setup_json['model']['gnn_edge_dim'],
        scattering_channels=setup_json['model']['scattering_channels'],
        scattering_dim=setup_json['model']['scattering_dim'],
        scattering_kernel_size=setup_json['model']['scattering_kernel_size'],
        scattering_stride=setup_json['model']['scattering_stride'],
        scattering_padding=setup_json['model']['scattering_padding'],
        composition_dim=setup_json['model']['composition_dim'],
        decoder_hidden_dim=setup_json['model']['decoder_hidden_dim'],
        position_output_dim=setup_json['model']['position_output_dim'],
        atom_output_dim=setup_json['model']['atom_output_dim'],
        cell_output_dim=setup_json['model']['cell_output_dim'],
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(setup_json['start_from_checkpoint']))
    
    # Load normalization parameters
    if setup_json['data']['normalize_cell_parameters']:
        cell_means = torch.tensor([
            setup_json['data']['cell_normalization']['a']['mean'],
            setup_json['data']['cell_normalization']['b']['mean'],
            setup_json['data']['cell_normalization']['c']['mean'],
            setup_json['data']['cell_normalization']['alpha']['mean'],
            setup_json['data']['cell_normalization']['beta']['mean'],
            setup_json['data']['cell_normalization']['gamma']['mean'],
        ]).float().to(device)
        cell_stds = torch.tensor([
            setup_json['data']['cell_normalization']['a']['std'],
            setup_json['data']['cell_normalization']['b']['std'],
            setup_json['data']['cell_normalization']['c']['std'],
            setup_json['data']['cell_normalization']['alpha']['std'],
            setup_json['data']['cell_normalization']['beta']['std'],
            setup_json['data']['cell_normalization']['gamma']['std'],
        ]).float().to(device)
    
    if setup_json['data']['normalize_atom_positions']:
        atom_position_means = torch.tensor(setup_json['data']['atom_position_normalization']['mean']).float().to(device)
        atom_position_stds = torch.tensor(setup_json['data']['atom_position_normalization']['std']).float().to(device)

    if setup_json['data']['normalize_distances']:
        distance_means = torch.tensor(setup_json['data']['distance_normalization']['mean']).float().to(device)
        distance_stds = torch.tensor(setup_json['data']['distance_normalization']['std']).float().to(device)

    beta = setup_json['training']['beta']
    out_dim = setup_json['model']['out_dim']
    
    # Results dict
    results = {
        'latent_point': [],
        'cell_parameters': [],
        'cell_positions': [],
        'cell_atoms': [],
        'cif_path': [],
        'pca_components': [],
    }
    
    # Inference
    model.eval()
    for batch in tqdm(interp_loader, desc='Interpolating', disable=setup_json['disable_tqdm']):
        this_batch_size = len(batch[0])
        interp_point, interp_index = batch
        interp_point = interp_point.to(device)
        interp_index = interp_index.to(device)
        composition = None
        
        with torch.no_grad():
            cell_parameters, cell_positions, cell_atoms = model.decode(
                interp_point, 
                composition=composition,
            )
        
        # Store latent points
        results['latent_point'].extend(interp_point.cpu().tolist())
        
        # Store predictions
        results['cell_parameters'].extend(cell_parameters.cpu().tolist())
        results['cell_positions'].extend(cell_positions.cpu().tolist())
        results['cell_atoms'].extend(torch.argmax(cell_atoms, dim=2).cpu().tolist())
        
        # Denormalize cell parameters
        if setup_json['data']['normalize_cell_parameters']:
            cell_parameters = (cell_parameters * cell_stds) + cell_means
        
        # Rounding positions to 5 decimals
        cell_positions = torch.round(cell_positions, decimals=5)
        
        # Create CIF files
        for batch_index in range(this_batch_size):
            # Prediction
            try:
                create_cif(
                    cell_params = cell_parameters[batch_index].detach().cpu().numpy(),
                    cell_positions = cell_positions[batch_index].detach().cpu().numpy(),
                    cell_atoms = cell_atoms[batch_index].detach().cpu().numpy(),
                    filename = f'{setup_json["model_root"]}{setup_json["experiment_name"]}/interpolation_predictions/sample{interp_index[batch_index]}',
                    prediction=True,
                    composition=None,
                    simplified_atom_identities=setup_json['training']['simplified_atom_identities'],
                )
            except:
                print(f'Failed to create CIF file for prediction of sample {interp_index[batch_index]}.')
    
    # Save results dictionary as json
    with open(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/interpolation_predictions/results.json', 'w') as f:
        json.dump(results, f)