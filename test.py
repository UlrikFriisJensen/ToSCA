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

#%% Suppress warnings
warnings.filterwarnings("ignore")

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
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(setup_json['start_from_checkpoint']))
    
    # Loss functions
    loss_fn_cell_parameters = torch.nn.MSELoss()
    loss_fn_cell_positions = torch.nn.MSELoss()
    loss_fn_cell_atoms = torch.nn.CrossEntropyLoss()
    # loss_fn_kld = torch.nn.KLDivLoss()
    
    beta = setup_json['training']['beta']
    
    #%% Test
    model.eval()
    test_loss = 0
    cell_parameters_loss = 0
    cell_positions_loss = 0
    cell_atoms_loss = 0
    kld_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Testing'):
            batch = batch.to(device)
            
            # Forward pass
            cell_parameters, cell_positions, cell_atoms, kld, post_mean, post_log_std, prior_mean, prior_log_std, z_sample = model.forward(
                x = torch.cat((batch.x, batch.pos_abs), dim=1), 
                edge_index = batch.edge_index, 
                scattering = batch.y['xPDF'][:,1,:].unsqueeze(-1),
                edge_attr = batch.edge_attr, 
                batch = batch.batch,
            )
            
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
            
            # Reshape atom predictions
            cell_atoms = cell_atoms.reshape(-1, cell_atoms.size(-1))
            cell_atoms_true = cell_atoms_true.reshape(-1).long()
            
            # Loss
            loss_cell_parameters = loss_fn_cell_parameters(cell_parameters, batch.y['cell_params'].view(-1, 6))
            loss_cell_positions = loss_fn_cell_positions(cell_positions, cell_positions_true)
            loss_cell_atoms = loss_fn_cell_atoms(cell_atoms, cell_atoms_true)
            loss_kld = kld.mean()
            
            total_loss = loss_cell_parameters + loss_cell_positions + loss_cell_atoms + (loss_kld * beta)
            
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
    ax.set_yscale('log')
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


