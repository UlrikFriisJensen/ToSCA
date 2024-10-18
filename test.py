#%% Imports
import argparse
import json
import warnings
import numpy as np
from modules.CHILI import CHILI
from modules.net import SCVAE
from torch_geometric.loader import DataLoader
import torch
import datetime
import pathlib
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ase import Atoms
from ase.io import write
from sklearn.decomposition import PCA
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
    parser.add_argument('--test_data', type=str, help='Whether to use the validation or test data')
    parser.add_argument('--setup_json', type=str, help='Path to the setup json file')
    args = parser.parse_args()
    
    # Load setup json
    with open(args.setup_json, 'r') as f:
        setup_json = json.load(f)
    
    #%% General setup
    
    # Make predictions folder
    predictions_folder = f'{setup_json["model_root"]}{setup_json["experiment_name"]}/predictions'
    ground_truth_folder = f'{setup_json["model_root"]}{setup_json["experiment_name"]}/ground_truth'
    pathlib.Path(predictions_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(ground_truth_folder).mkdir(parents=True, exist_ok=True)

    # Set random seed
    torch.manual_seed(setup_json['random_seed'])
    np.random.seed(setup_json['random_seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CHILI dataset
    dataset = CHILI(
        root=setup_json['data']['root'],
        dataset=setup_json['data']['name']
    )
    
    # Load data splits
    dataset.load_data_split(
        split_strategy=setup_json['data']['split_strategy'],
        stratify_on=setup_json['data']['stratify_column'],
    )
    
    # Dataloader
    if args.test_data == 'validation':
        data_loader = DataLoader(dataset.validation_set, batch_size=setup_json['data']['batch_size'], shuffle=False)
    elif args.test_data == 'test':
        data_loader = DataLoader(dataset.test_set, batch_size=setup_json['data']['batch_size'], shuffle=False)
    
    # Load model
    model = SCVAE(
        latent_dim=setup_json['model']['latent_dim'],
        out_dim=setup_json['model']['out_dim'],
        gnn_dim=setup_json['model']['gnn_dim'],
        gnn_heads=setup_json['model']['gnn_heads'],
        gnn_edge_dim=setup_json['model']['gnn_edge_dim'],
        scattering_channels=setup_json['model']['scattering_channels'],
        scattering_dim=setup_json['model']['scattering_dim'],
        scattering_kernel_size=setup_json['model']['scattering_kernel_size'],
        scattering_stride=setup_json['model']['scattering_stride'],
        scattering_padding=setup_json['model']['scattering_padding'],
        decoder_hidden_dim=setup_json['model']['decoder_hidden_dim'],
        position_output_dim=setup_json['model']['position_output_dim'],
        atom_output_dim=setup_json['model']['atom_output_dim'],
        cell_output_dim=setup_json['model']['cell_output_dim'],
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(setup_json['start_from_checkpoint']))
    
    # Loss functions
    loss_fn_cell_parameters = torch.nn.MSELoss()
    # loss_fn_cell_positions = torch.nn.MSELoss()
    # loss_fn_cell_atoms = torch.nn.CrossEntropyLoss()
    # loss_fn_kld = torch.nn.KLDivLoss()
    loss_fn_cell_positions = weighted_MSELoss()
    loss_fn_cell_atoms = weighted_CrossEntropyLoss()
    
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
    
    #%% Test
    model.eval()
    test_loss = 0
    cell_parameters_loss = 0
    cell_positions_loss = 0
    cell_atoms_loss = 0
    kld_loss = 0
    
    latent_space_means = []
    sample_crystal_types = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Testing', disable=setup_json['disable_tqdm']):
            # Put batch on device
            batch = batch.to(device)
            
            # Normalize scattering
            batch_scattering = batch.y['xPDF'][:,1,:].unsqueeze(-1)
            if setup_json['data']['normalize_scattering']:
                # Normalize so highest peak in each sample is 1
                # batch_scattering -= torch.amin(batch_scattering, dim=1, keepdim=True)[0]
                batch_scattering /= torch.amax(batch_scattering, dim=1, keepdim=True)[0]

            # Normalize cell parameters
            cell_parameters_true = batch.y['cell_params'].view(-1, 6)
            if setup_json['data']['normalize_cell_parameters']:
                cell_parameters_true = (cell_parameters_true - cell_means) / cell_stds
            cell_parameters_true = cell_parameters_true.float()
            
            # Normalize atom positions
            batch_positions = batch.pos_abs
            if setup_json['data']['normalize_atom_positions']:
                batch_positions = (batch_positions - atom_position_means) / atom_position_stds
            batch_positions = batch_positions.float()

            # Normalize distances
            batch_distances = batch.edge_attr
            if setup_json['data']['normalize_distances']:
                batch_distances = (batch_distances - distance_means) / distance_stds
            batch_distances = batch_distances.float()
            
            # Forward pass
            cell_parameters, cell_positions, cell_atoms, kld, post_mean, post_log_std, prior_mean, prior_log_std, z_sample = model.forward(
                x = torch.cat((batch.x, batch.pos_abs), dim=1), 
                edge_index = batch.edge_index, 
                scattering = batch.y['xPDF'][:,1,:].unsqueeze(-1),
                edge_attr = batch.edge_attr, 
                batch = batch.batch,
            )
            
            # Store latent space means
            latent_space_means.extend(z_sample.cpu().numpy())
            sample_crystal_types.extend(batch.y['crystal_type'])

            # Assign batch labels to unit cell positions
            unit_cell_batch = torch.zeros(batch.y['unit_cell_pos_frac'].shape[0], dtype=torch.long)
            index_sum = 0
            for i, unit_cell_atoms in enumerate(batch.y['unit_cell_n_atoms']):
                unit_cell_batch[index_sum:index_sum + unit_cell_atoms] = i
                index_sum += unit_cell_atoms
            
            cell_positions_true = torch.zeros_like(cell_positions).to(device) - 1
            cell_atoms_true = torch.zeros(cell_atoms.size(0), cell_atoms.size(1)).to(device)
            
            for batch_index, unit_cell_size in enumerate(batch.y['unit_cell_n_atoms']):
                cell_positions_true[batch_index, :unit_cell_size] = batch.y['unit_cell_pos_frac'][unit_cell_batch == batch_index]
                cell_atoms_true[batch_index, :unit_cell_size] = batch.y['unit_cell_x'][unit_cell_batch == batch_index, 0]


            # Denormalize cell parameters
            cell_parameters_pred = cell_parameters
            if setup_json['data']['normalize_cell_parameters']:
                cell_parameters_pred = (cell_parameters_pred * cell_stds) + cell_means

            # Create CIF files
            for batch_index in range(len(batch)):
                # Ground truth
                ground_truth_composition = create_cif(
                    cell_params = batch.y['cell_params'].view(-1, 6)[batch_index].detach().cpu().numpy(),
                    cell_positions = cell_positions_true[batch_index].detach().cpu().numpy(),
                    cell_atoms = cell_atoms_true[batch_index].detach().cpu().numpy(),
                    filename = f'{setup_json["model_root"]}{setup_json["experiment_name"]}/ground_truth/{batch.y["crystal_type"][batch_index]}',
                    prediction=False
                )

                # Prediction
                try:
                    create_cif(
                        cell_params = cell_parameters_pred[batch_index].detach().cpu().numpy(),
                        cell_positions = cell_positions[batch_index].detach().cpu().numpy(),
                        cell_atoms = cell_atoms[batch_index].detach().cpu().numpy(),
                        filename = f'{setup_json["model_root"]}{setup_json["experiment_name"]}/predictions/{batch.y["crystal_type"][batch_index]}',
                        prediction=True,
                        composition=ground_truth_composition,
                        simplified_atom_identities=setup_json['training']['simplified_atom_identities'],
                    )
                except:
                    print(f'Failed to create CIF file for prediction of {ground_truth_composition} as a {batch.y["crystal_type"][batch_index]} structure')

            # Reshape atom predictions
            cell_atoms = cell_atoms.reshape(-1, cell_atoms.size(-1))
            cell_atoms_true = cell_atoms_true.reshape(-1).long()
            
            # Make loss weights
            cell_positions_weights = torch.where(cell_positions_true != -1, 1, 0).float().to(device)
            cell_atoms_weights = torch.where(cell_atoms_true != 0, 1, 0.1).float().to(device)
            
            # Simplify atom identities
            if setup_json['training']['simplified_atom_identities']:
                # Map atom number 0 to logit 0 (No atom)
                cell_atoms_true = torch.where(cell_atoms_true == 0, 0, cell_atoms_true)
                # Map atom numbers of ligands to logit 1 (Ligand) # [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53]
                for ligand in setup_json['training']['ligands']:
                    cell_atoms_true = torch.where(cell_atoms_true == ligand, 1, cell_atoms_true)
                # Map all other atom numbers to logit 2 (Metal)
                cell_atoms_true = torch.where(cell_atoms_true >= 2, 2, cell_atoms_true)
            
            # Loss
            loss_cell_parameters = loss_fn_cell_parameters(cell_parameters, cell_parameters_true) 
            
            # loss_cell_positions = loss_fn_cell_positions(cell_positions, cell_positions_true) # Unweighted
            loss_cell_positions = loss_fn_cell_positions(cell_positions, cell_positions_true, cell_positions_weights) # Weighted
            
            # loss_cell_atoms = loss_fn_cell_atoms(cell_atoms, cell_atoms_true) # Unweighted
            loss_cell_atoms = loss_fn_cell_atoms(cell_atoms, cell_atoms_true, cell_atoms_weights) # Weighted
            
            loss_kld = kld.mean()
            
            total_loss = torch.log(loss_cell_parameters + loss_cell_positions + loss_cell_atoms) + (loss_kld * beta)
            
            # Store loss
            test_loss += total_loss.item()
            cell_parameters_loss += loss_cell_parameters.item()
            cell_positions_loss += loss_cell_positions.item()
            cell_atoms_loss += loss_cell_atoms.item()
            kld_loss += loss_kld.item()
            
        test_loss /= len(data_loader)
        cell_parameters_loss /= len(data_loader)
        cell_positions_loss /= len(data_loader)
        cell_atoms_loss /= len(data_loader)
        kld_loss /= len(data_loader)
        
        print(f'Test loss: {test_loss:.4f}')
        print(f'Cell parameters loss: {cell_parameters_loss:.4f}')
        print(f'Cell positions loss: {cell_positions_loss:.4f}')
        print(f'Cell atoms loss: {cell_atoms_loss:.4f}')
        print(f'KLD loss: {kld_loss:.4f}')
        
        # Save loss
        with open(f'{setup_json["model_root"]}/{setup_json["experiment_name"]}/{args.test_data}_loss.txt', 'w') as f:
            f.write(f'Test loss: {test_loss:.6f}\n')
            f.write(f'Cell parameters loss: {cell_parameters_loss:.6f}\n')
            f.write(f'Cell positions loss: {cell_positions_loss:.6f}\n')
            f.write(f'Cell atoms loss: {cell_atoms_loss:.6f}\n')
            f.write(f'KLD loss: {kld_loss:.6f}\n')

    #%% Plot latent space

    latent_space_means = np.array(latent_space_means)
    print(latent_space_means.shape)
    print(len(sample_crystal_types))

    # Reduce dimensions with PCA
    pca = PCA(n_components=2)
    latent_space_pca = pca.fit_transform(latent_space_means)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.scatterplot(x=latent_space_pca[:,0], y=latent_space_pca[:,1], hue=sample_crystal_types, ax=ax)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    fig.tight_layout()
    fig.savefig(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/latent_space.png', dpi=300)
    
    #%% Plot loss curves
    
    # Load loss data
    loss_data = pd.read_csv(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/training_log.csv', sep=',')

    # Plot loss curves

    # Total loss
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(loss_data['epoch'], loss_data['train_loss'], label='Train loss')
    ax.plot(loss_data['epoch'], loss_data['validation_loss'], label='Validation loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    #ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/loss_curve.png', dpi=300)

    # Train loss components
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(loss_data['epoch'], loss_data['train_loss'], label='Total')
    ax.plot(loss_data['epoch'], loss_data['train_loss_cell_parameters'], label='Cell parameters')
    ax.plot(loss_data['epoch'], loss_data['train_loss_cell_positions'], label='Cell positions')
    ax.plot(loss_data['epoch'], loss_data['train_loss_cell_atoms'], label='Cell atoms')
    ax.plot(loss_data['epoch'], loss_data['train_loss_kld'] * beta, label='KLD')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/train_loss_components.png', dpi=300)

    # Validation loss components
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(loss_data['epoch'], loss_data['validation_loss'], label='Total')
    ax.plot(loss_data['epoch'], loss_data['validation_loss_cell_parameters'], label='Cell parameters')
    ax.plot(loss_data['epoch'], loss_data['validation_loss_cell_positions'], label='Cell positions')
    ax.plot(loss_data['epoch'], loss_data['validation_loss_cell_atoms'], label='Cell atoms')
    ax.plot(loss_data['epoch'], loss_data['validation_loss_kld'] * beta, label='KLD')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/validation_loss_components.png', dpi=300)


