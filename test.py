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
        dataset=setup_json['data']['name'],
        graph_type=setup_json['data']['graph_type'],
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
        seperate_decoder=setup_json['training']['seperate_decoder'],
    ).to(device)
    
    model_path = f'{setup_json["model_root"]}{setup_json["experiment_name"]}/'
    if setup_json['training']['beta_annealing']:
        model_names = ['best_model.pth', 'best_model_annealed.pth']
    else:
        model_names = ['best_model.pth']
    
    for model_name in model_names:
        # Load model weights
        model.load_state_dict(torch.load(model_path + model_name))
        
        # Make predictions folder
        predictions_folder = f'{model_path}/predictions_{model_name[:-4]}'
        ground_truth_folder = f'{model_path}/ground_truth'
        pathlib.Path(predictions_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(ground_truth_folder).mkdir(parents=True, exist_ok=True)
        
        # Loss functions
        loss_fn_cell_parameters = torch.nn.MSELoss()
        # loss_fn_cell_positions = torch.nn.MSELoss()
        # loss_fn_cell_atoms = torch.nn.CrossEntropyLoss()
        # loss_fn_kld = torch.nn.KLDivLoss()
        loss_fn_cell_positions = weighted_MSELoss()
        loss_fn_cell_atoms = weighted_CrossEntropyLoss()
        loss_fn_latent_mean = torch.nn.MSELoss()
        loss_fn_latent_std = torch.nn.MSELoss()
        
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
        
        #%% Test on validation/test set
        model.eval()
        test_loss = 0
        test_reconstruction_loss = 0
        cell_parameters_loss = 0
        cell_positions_loss = 0
        cell_atoms_loss = 0
        kld_loss = 0
        
        ls_mean_posterior = []
        ls_std_posterior = []
        ls_mean_prior = []
        ls_std_prior = []
        ls_sample = []
        sample_crystal_types = []

        if setup_json['data']['name'] in ['CHILI-3K', 'CHILI-Interpolation', 'CHILI-Interpolation_v2']:
            # Logging for analysis of Crystal type dependent performance
            
            # Crystal type dependent losses
            loss_CrystalType = {'total': [], 'reconstruction_loss': [], 'cell_parameters': [], 'cell_positions': [], 'cell_atoms': [], 'kld': [], 'crystalType': [], 'particleSize': []}
            
            # Crystal type dependent reconstructions
            reconstructions_CrystalType = {'crystalType': [], 'n_atoms': [], 'n_oxygens': [], 'n_metals': [], 'cell_parameters': [], 'cell_positions': [], 'cell_atoms': [], 'latent_space_mean': [], 'latent_space_std': [], 'latent_space_mean_prior': [], 'latent_space_std_prior': [], "true_cell_parameters": [], "true_cell_positions": [], "true_cell_atoms": []}
            
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Testing', disable=setup_json['disable_tqdm']):
                # Put batch on device
                batch = batch.to(device)
                this_batch_size = batch.batch.amax().item() + 1
                
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
                
                # Node features
                batch_features = torch.cat((batch.x, batch_positions), dim=1)
                
                # Composition conditioning
                composition = torch.zeros(this_batch_size, 119).to(device)
                elements_in_batch = batch.y['atomic_species'].long()
                index_counter = 0
                for i in range(this_batch_size):
                    n_elements = batch.y['n_atomic_species'][i]
                    composition[i, elements_in_batch[index_counter:index_counter + n_elements]] = 1
                    index_counter += n_elements
                composition[:, 0] = 1 

                # Forward pass
                cell_parameters, cell_positions, cell_atoms, kld, post_mean, post_log_std, prior_mean, prior_log_std, z_sample = model.forward(
                    x = batch_features, 
                    edge_index = batch.edge_index, 
                    scattering = batch_scattering,
                    composition=composition,
                    edge_attr = batch_distances, 
                    batch = batch.batch,
                )
                
                # Predict using the prior
                prior_cell_parameters, prior_cell_positions, prior_cell_atoms, _, _, _ = model.predict(
                    scattering=batch_scattering,
                    composition=composition,
                )
                
                # Store latent space means
                ls_mean_posterior.extend(post_mean.cpu().tolist())
                ls_std_posterior.extend(post_log_std.exp().cpu().tolist())
                ls_mean_prior.extend(prior_mean.cpu().tolist())
                ls_std_prior.extend(prior_log_std.exp().cpu().tolist())
                ls_sample.extend(z_sample.cpu().tolist())
                sample_crystal_types.extend(batch.y['crystal_type'])

                if setup_json['data']['graph_type'] in ['unit_cell', 'super_cell']:
                    cell_positions_true = batch.pos_frac
                    cell_positions_true = cell_positions_true.reshape(this_batch_size, out_dim, -1)
                    cell_atoms_true = batch.x[:,0]
                    cell_atoms_true = cell_atoms_true.reshape(this_batch_size, out_dim).long()
                elif setup_json['data']['graph_type'] == 'combi':
                    cell_positions_true = batch.pos_frac_target
                    cell_positions_true = cell_positions_true.reshape(this_batch_size, out_dim, -1)
                    cell_atoms_true = batch.x_target[:,0]
                    cell_atoms_true = cell_atoms_true.reshape(this_batch_size, out_dim).long()
                elif setup_json['data']['graph_type'] == 'central-target':
                    cell_positions_true = batch.pos_frac
                    cell_atoms_true = batch.x[:,0]
                    
                    batch_indices = batch.batch
                    
                    target_size = out_dim * this_batch_size
                    
                    if cell_atoms_true.size(0) < target_size:
                        cell_positions_true_padded = torch.zeros_like(cell_positions).to(device) - 1
                        cell_atoms_true_padded = torch.zeros(cell_atoms.size(0), cell_atoms.size(1)).to(device)
                        
                        for i in range(this_batch_size):
                            batch_index_mask = batch_indices == i
                            cell_positions_true_padded[i, :sum(batch_index_mask)] = cell_positions_true[batch_index_mask]
                            cell_atoms_true_padded[i, :sum(batch_index_mask)] = cell_atoms_true[batch_index_mask]
                            
                        cell_positions_true = cell_positions_true_padded
                        cell_atoms_true = cell_atoms_true_padded
                    elif cell_atoms_true.size(0) > target_size:
                        raise ValueError('Number of atoms in central target graph is larger than expected')
                    else:
                        cell_positions_true = cell_positions_true.reshape(this_batch_size, out_dim, -1)
                        cell_atoms_true = cell_atoms_true.reshape(this_batch_size, out_dim).long()
                else:
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
                prior_cell_parameters_pred = prior_cell_parameters
                if setup_json['data']['normalize_cell_parameters']:
                    cell_parameters_pred = (cell_parameters_pred * cell_stds) + cell_means
                    prior_cell_parameters_pred = (prior_cell_parameters_pred * cell_stds) + cell_means
                
                # Rounding positions to 5 decimals
                cell_positions = torch.round(cell_positions, decimals=5)
                
                # Create CIF files
                for batch_index in range(this_batch_size):
                    # Crystal type dependent reconstructions
                    if setup_json['data']['name'] in ['CHILI-3K', 'CHILI-Interpolation', 'CHILI-Interpolation_v2']:
                        ct_cell_atoms_true = cell_atoms_true[batch_index]
                        # Simplify atom identities
                        if setup_json['training']['simplified_atom_identities']:
                            # Map atom number 0 to logit 0 (No atom)
                            ct_cell_atoms_true = torch.where(ct_cell_atoms_true == 0, 0, ct_cell_atoms_true)
                            # Map atom numbers of ligands to logit 1 (Ligand) # [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53]
                            for ligand in setup_json['training']['ligands']:
                                ct_cell_atoms_true = torch.where(ct_cell_atoms_true == ligand, 1, ct_cell_atoms_true)
                            # Map all other atom numbers to logit 2 (Metal)
                            ct_cell_atoms_true = torch.where(ct_cell_atoms_true >= 2, 2, ct_cell_atoms_true)
                        
                        # Make loss weights
                        ct_cell_positions_weights = torch.where(cell_positions_true[batch_index] != -1, 1, 0).float().to(device)
                        ct_cell_atoms_weights = torch.where(cell_atoms_true[batch_index] != 0, 1, 0.1).float().to(device)
                        
                        # Find argmax of atoms
                        cell_atoms_rec = torch.argmax(cell_atoms[batch_index], dim=1)
                        
                        # Remove atoms with atom number 0
                        cell_positions_rec = cell_positions[batch_index][cell_atoms_rec != 0]
                        cell_atoms_rec = cell_atoms_rec[cell_atoms_rec != 0]
                        
                        # Calculate loss
                        ct_loss_cell_parameters = loss_fn_cell_parameters(cell_parameters[batch_index], cell_parameters_true[batch_index])
                        ct_loss_cell_positions = loss_fn_cell_positions(cell_positions[batch_index], cell_positions_true[batch_index], ct_cell_positions_weights)
                        ct_loss_cell_atoms = loss_fn_cell_atoms(cell_atoms[batch_index], ct_cell_atoms_true.long(), ct_cell_atoms_weights)
                        ct_loss_rec = ct_loss_cell_parameters + ct_loss_cell_positions + ct_loss_cell_atoms
                            
                        ct_loss_kld = kld[batch_index].mean()
                        if setup_json['training']['latent_mse']:
                            ct_loss_latent_mean = loss_fn_latent_mean(prior_mean[batch_index], post_mean[batch_index])
                            ct_loss_latent_std = loss_fn_latent_std(prior_log_std[batch_index], post_log_std[batch_index])
                            ct_loss_latent = ct_loss_kld + ct_loss_latent_mean + ct_loss_latent_std
                        else:
                            ct_loss_latent = ct_loss_kld
                            
                        ct_total_loss = torch.log(ct_loss_rec + ((ct_loss_latent) * beta)) #torch.log(ct_loss_rec) + (ct_loss_kld * beta)
                        
                        # Log the Crystal type dependent losses
                        loss_CrystalType['total'].append(ct_total_loss.item())
                        loss_CrystalType['reconstruction_loss'].append(ct_loss_rec.item())
                        loss_CrystalType['cell_parameters'].append(ct_loss_cell_parameters.item())
                        loss_CrystalType['cell_positions'].append(ct_loss_cell_positions.item())
                        loss_CrystalType['cell_atoms'].append(ct_loss_cell_atoms.item())
                        loss_CrystalType['kld'].append(ct_loss_kld.item())
                        loss_CrystalType['crystalType'].append(batch.y['crystal_type'][batch_index])
                        loss_CrystalType['particleSize'].append(batch.y['np_size'][batch_index].item())
                        
                        # Log the Crystal type dependent reconstructions
                        reconstructions_CrystalType['crystalType'].append(batch.y['crystal_type'][batch_index])
                        reconstructions_CrystalType['n_atoms'].append(len(cell_atoms_rec))
                        reconstructions_CrystalType['n_oxygens'].append(torch.sum(cell_atoms_rec == 1).item())
                        reconstructions_CrystalType['n_metals'].append(torch.sum(cell_atoms_rec == 2).item())
                        reconstructions_CrystalType['cell_parameters'].append(cell_parameters_pred[batch_index].detach().cpu().tolist())
                        reconstructions_CrystalType['cell_positions'].append(cell_positions_rec.detach().cpu().tolist())
                        reconstructions_CrystalType['cell_atoms'].append(cell_atoms_rec.detach().cpu().tolist())
                        reconstructions_CrystalType['latent_space_mean'].append(post_mean[batch_index].detach().cpu().tolist())
                        reconstructions_CrystalType['latent_space_std'].append(post_log_std[batch_index].detach().cpu().tolist())
                        reconstructions_CrystalType['latent_space_mean_prior'].append(prior_mean[batch_index].detach().cpu().tolist())
                        reconstructions_CrystalType['latent_space_std_prior'].append(prior_log_std[batch_index].detach().cpu().tolist())
                        reconstructions_CrystalType['true_cell_parameters'].append(cell_parameters_true[batch_index].detach().cpu().tolist())
                        reconstructions_CrystalType['true_cell_positions'].append(cell_positions_true[batch_index].detach().cpu().tolist())
                        reconstructions_CrystalType['true_cell_atoms'].append(cell_atoms_true[batch_index].detach().cpu().tolist())
                    
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
                            filename = f'{setup_json["model_root"]}{setup_json["experiment_name"]}/predictions_{model_name[:-4]}/{batch.y["crystal_type"][batch_index]}',
                            prediction=True,
                            composition=ground_truth_composition,
                            simplified_atom_identities=setup_json['training']['simplified_atom_identities'],
                        )
                    except:
                        print(f'Failed to create CIF file for prediction of {ground_truth_composition} as a {batch.y["crystal_type"][batch_index]} structure')

                    # Prior prediction
                    try:
                        create_cif(
                            cell_params = prior_cell_parameters_pred[batch_index].detach().cpu().numpy(),
                            cell_positions = prior_cell_positions[batch_index].detach().cpu().numpy(),
                            cell_atoms = prior_cell_atoms[batch_index].detach().cpu().numpy(),
                            filename = f'{setup_json["model_root"]}{setup_json["experiment_name"]}/predictions_{model_name[:-4]}/{batch.y["crystal_type"][batch_index]}',
                            prediction=True,
                            composition=ground_truth_composition + '_prior',
                            simplified_atom_identities=setup_json['training']['simplified_atom_identities'],
                        )
                    except:
                        print(f'Failed to create CIF file for prior prediction of {ground_truth_composition} as a {batch.y["crystal_type"][batch_index]} structure')
                # Free up memory related to batch
                del batch, batch_scattering, batch_positions, batch_distances, batch_features
                
                # Reshape predictions
                cell_positions = cell_positions.reshape(this_batch_size, out_dim, -1)
                cell_positions_true = cell_positions_true.reshape(this_batch_size, out_dim, -1)
                
                cell_atoms = cell_atoms.reshape(-1, setup_json['model']['atom_output_dim'])
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
                
                # Loss calculations

                # Reconstruction loss
                loss_cell_parameters = loss_fn_cell_parameters(cell_parameters, cell_parameters_true) 
                
                # loss_cell_positions = loss_fn_cell_positions(cell_positions, cell_positions_true) # Unweighted
                loss_cell_positions = loss_fn_cell_positions(cell_positions, cell_positions_true, cell_positions_weights) # Weighted
                
                # loss_cell_atoms = loss_fn_cell_atoms(cell_atoms, cell_atoms_true) # Unweighted
                loss_cell_atoms = loss_fn_cell_atoms(cell_atoms, cell_atoms_true, cell_atoms_weights) # Weighted
                
                loss_reconstruction = loss_cell_parameters + loss_cell_positions + loss_cell_atoms

                # Latent loss
                loss_kld = kld.mean()
                if setup_json['training']['latent_mse']:
                    loss_latent_mean = loss_fn_latent_mean(prior_mean, post_mean)
                    loss_latent_std = loss_fn_latent_std(prior_log_std, post_log_std)
                    latent_loss = loss_kld + loss_latent_mean + loss_latent_std
                else:
                    latent_loss = loss_kld
                
                total_loss = torch.log(loss_reconstruction + ((latent_loss) * beta)) #torch.log(loss_reconstruction) + (loss_kld * beta)
                
                # Store loss
                test_loss += total_loss.item()
                test_reconstruction_loss += loss_reconstruction.item()
                cell_parameters_loss += loss_cell_parameters.item()
                cell_positions_loss += loss_cell_positions.item()
                cell_atoms_loss += loss_cell_atoms.item()
                kld_loss += loss_kld.item()
                    
                
            test_loss /= len(data_loader)
            test_reconstruction_loss /= len(data_loader)
            cell_parameters_loss /= len(data_loader)
            cell_positions_loss /= len(data_loader)
            cell_atoms_loss /= len(data_loader)
            kld_loss /= len(data_loader)
            
            print(f'Test loss: {test_loss:.4f}')
            print(f'Reconstruction loss: {test_reconstruction_loss:.4f}')
            print(f'Cell parameters loss: {cell_parameters_loss:.4f}')
            print(f'Cell positions loss: {cell_positions_loss:.4f}')
            print(f'Cell atoms loss: {cell_atoms_loss:.4f}')
            print(f'KLD loss: {kld_loss:.4f}')
            
            # Save loss
            with open(f'{setup_json["model_root"]}/{setup_json["experiment_name"]}/{args.test_data}_loss_{model_name[:-4]}.txt', 'w') as f:
                f.write(f'Test loss: {test_loss:.6f}\n')
                f.write(f'Reconstruction loss: {test_reconstruction_loss:.6f}\n')
                f.write(f'Cell parameters loss: {cell_parameters_loss:.6f}\n')
                f.write(f'Cell positions loss: {cell_positions_loss:.6f}\n')
                f.write(f'Cell atoms loss: {cell_atoms_loss:.6f}\n')
                f.write(f'KLD loss: {kld_loss:.6f}\n')
                
            # Saving crystal type dependent information
            if setup_json['data']['name'] in ['CHILI-3K', 'CHILI-Interpolation', 'CHILI-Interpolation_v2']:
                # Make folders
                pathlib.Path(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/CrystalTypeAnalysis').mkdir(parents=True, exist_ok=True)
                
                # Save loss dicts as json
                with open(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/CrystalTypeAnalysis/losses_{model_name[:-4]}.json', 'w') as f:
                    json.dump(loss_CrystalType, f)
                    
                # Save reconstructions
                with open(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/CrystalTypeAnalysis/reconstructions_{model_name[:-4]}.json', 'w') as f:
                    json.dump(reconstructions_CrystalType, f)
                

        #%% Plot latent space

        ls_mean_posterior = np.array(ls_mean_posterior)
        ls_std_posterior = np.array(ls_std_posterior)
        ls_mean_prior = np.array(ls_mean_prior)
        ls_std_prior = np.array(ls_std_prior)
        ls_sample = np.array(ls_sample)

        if setup_json['model']['latent_dim'] == 2:
            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            sns.scatterplot(x=ls_mean_posterior[:,0], y=ls_mean_posterior[:,1], hue=sample_crystal_types, style=sample_crystal_types, ax=ax, palette='tab20')
            ax.set_xlabel('Latent dim 1')
            ax.set_ylabel('Latent dim 2')
            fig.tight_layout()
            fig.savefig(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/latent_space_posterior_{model_name[:-4]}.png', dpi=300)
            
            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            sns.scatterplot(x=ls_mean_prior[:,0], y=ls_mean_prior[:,1], hue=sample_crystal_types, style=sample_crystal_types, ax=ax, palette='tab20')
            ax.set_xlabel('Latent dim 1')
            ax.set_ylabel('Latent dim 2')
            fig.tight_layout()
            fig.savefig(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/latent_space_prior_{model_name[:-4]}.png', dpi=300)
            
        elif setup_json['model']['latent_dim'] > 2:
            # Reduce dimensions with PCA
            pca = PCA(n_components=2)
            latent_space_2d_pca_posterior = pca.fit_transform(ls_mean_posterior)
            
            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            sns.scatterplot(x=latent_space_2d_pca_posterior[:,0], y=latent_space_2d_pca_posterior[:,1], hue=sample_crystal_types, style=sample_crystal_types, ax=ax, palette='tab20')
            ax.set_xlabel('PC 1')
            ax.set_ylabel('PC 2')
            fig.tight_layout()
            fig.savefig(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/latent_space_pca_posterior_{model_name[:-4]}.png', dpi=300)
            
            if (sum(abs(ls_mean_prior)) > 1e-3).all():
                latent_space_2d_pca_prior = pca.transform(ls_mean_prior)
            
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                sns.scatterplot(x=latent_space_2d_pca_prior[:,0], y=latent_space_2d_pca_prior[:,1], hue=sample_crystal_types, style=sample_crystal_types, ax=ax, palette='tab20')
                ax.set_xlabel('PC 1')
                ax.set_ylabel('PC 2')
                fig.tight_layout()
                fig.savefig(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/latent_space_pca_prior_{model_name[:-4]}.png', dpi=300)
            
            # Reduce dimensions with t-SNE
            tsne = TSNE(n_components=2, random_state=setup_json['random_seed'])
            latent_space_2d_tsne_posterior = tsne.fit_transform(ls_mean_posterior)
            
            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            sns.scatterplot(x=latent_space_2d_tsne_posterior[:,0], y=latent_space_2d_tsne_posterior[:,1], hue=sample_crystal_types, style=sample_crystal_types, ax=ax, palette='tab20')
            ax.set_xlabel('t-SNE dim 1')
            ax.set_ylabel('t-SNE dim 2')
            fig.tight_layout()
            fig.savefig(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/latent_space_tsne_posterior_{model_name[:-4]}.png', dpi=300)
            
            if (sum(abs(ls_mean_prior)) > 1e-3).all():
                latent_space_2d_tsne_prior = tsne.fit_transform(ls_mean_prior)
            
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                sns.scatterplot(x=latent_space_2d_tsne_prior[:,0], y=latent_space_2d_tsne_prior[:,1], hue=sample_crystal_types, style=sample_crystal_types, ax=ax, palette='tab20')
                ax.set_xlabel('t-SNE dim 1')
                ax.set_ylabel('t-SNE dim 2')
                fig.tight_layout()
                fig.savefig(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/latent_space_tsne_prior_{model_name[:-4]}.png', dpi=300)
    
    
    #%% Plot loss curves
    
    # Load loss data
    loss_data = pd.read_csv(f'{setup_json["model_root"]}{setup_json["experiment_name"]}/training_log.csv', sep=',')

    finetuning_data = loss_data[loss_data['stage'] == 'Fine-tuning']
    loss_data = loss_data[loss_data['stage'] == 'Training']
    
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
    ax.plot(loss_data['epoch'], loss_data['train_loss_reconstruction'], label='Reconstruction')
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
    ax.plot(loss_data['epoch'], loss_data['validation_loss_reconstruction'], label='Reconstruction')
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
    