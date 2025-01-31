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
    parser.add_argument('--data_folder', type=str, help='Path to the folder with the experimental data')
    parser.add_argument('--setup_json', type=str, help='Path to the setup json file')
    args = parser.parse_args()
    
    # Load setup json
    with open(args.setup_json, 'r') as f:
        setup_json = json.load(f)
        
    # Make predictions folder
    predictions_folder = f'{setup_json["model_root"]}{setup_json["experiment_name"]}/experimental_predictions'
    pathlib.Path(predictions_folder).mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load experimental data
    data_paths = [str(p) for p in pathlib.Path(args.data_folder).glob('*.gr')]
    
    data_filepath = []
    data_composition = []
    data_pdf = []
    for data_path in data_paths:
        with open(data_path, 'r') as f:
            # Load data
            line_counter = 0
            for line in f:
                if line.startswith('composition'):
                    composition = line.split(' ')[-1]
                if line.startswith('0'):
                    header_line = line_counter
                    break
                line_counter += 1
        # Remove stochiometry from composition
        composition = re.sub(r'[0-9\.]+', '', composition)
        # Remove line breaks
        composition = composition.replace('\n', '')
        # # Split string on capital letters
        composition = re.findall('[A-Z][^A-Z]*', composition)

        # Translate composition to atom numbers
        composition = Atoms(symbols=composition).get_atomic_numbers()
        
        composition_onehot = np.zeros(len(119))
        composition_onehot[composition] = 1
        
        
        # Load data
        data = pd.read_csv(data_path, sep=' ', skiprows=header_line, names=['r [Å]', 'G(r) [Å⁻²]'])
        
        data_r = np.arange(0,60,0.01)
        data_Gr = np.interp(data_r, data['r [Å]'], data['G(r) [Å⁻²]'], left=0, right=0)
        data_Gr = data_Gr / np.amax(data_Gr)
        
        data_filepath.append(data_path)
        data_composition.append(composition_onehot)
        data_pdf.append(data_Gr)
    
    # Convert to tensors
    data_composition = torch.tensor(data_composition, dtype=torch.long)
    data_pdf = torch.tensor(data_pdf, dtype=torch.float32)
    
    exp_data = TensorDataset(data_pdf, data_composition)
    
    # Dataloader
    exp_loader = DataLoader(exp_data, batch_size=10, shuffle=False)
    
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
    
    model.eval()
    
    # Inference
    for batch in tqdm(exp_loader, desc='Inference', disable=setup_json['disable_tqdm']):
        pdf, composition = batch
        pdf = pdf.to(device)
        composition = composition.to(device)
        
        with torch.no_grad():
            cell_parameters, cell_positions, cell_atoms, prior_mean, prior_log_std, z_sample = model.predict(
                pdf, 
                composition
            )
        
        
        
    