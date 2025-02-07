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
from modules.loss_functions import weighted_MSELoss, weighted_CrossEntropyLoss
import torch.nn.functional as F

#%% Suppress warnings
warnings.filterwarnings("ignore")

#%% Main

if __name__ == "__main__":
    #%% Parse arguments
    parser = argparse.ArgumentParser(description='Train the SCVAE model')
    parser.add_argument('--setup_json', type=str, help='Path to the setup json file')
    args = parser.parse_args()
    
    # Load setup json
    with open(args.setup_json, 'r') as f:
        setup_json = json.load(f)
    
    #%% General setup
    
    # Make experiment folder
    experiment_folder = f'{setup_json["model_root"]}{setup_json["experiment_name"]}'
    pathlib.Path(experiment_folder).mkdir(parents=True, exist_ok=True)
    
    # Record date and time
    setup_json['experiment_start'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Set random seed
    torch.manual_seed(setup_json['random_seed'])
    np.random.seed(setup_json['random_seed'])
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #%% Prepare data

    # Load CHILI dataset
    dataset = CHILI(
        root=setup_json['data']['root'],
        dataset=setup_json['data']['name'],
        graph_type=setup_json['data']['graph_type'],
    )
    
    # Load/create data splits
    try:
        # Load existing data split
        dataset.load_data_split(
            split_strategy=setup_json['data']['split_strategy'],
            stratify_on=setup_json['data']['stratify_column'],
        )
    except FileNotFoundError:
        # Create new data split
        dataset.create_data_split(
            test_size=setup_json['data']['split']['test'],
            validation_size=setup_json['data']['split']['validation'],
            split_strategy=setup_json['data']['split_strategy'],
            stratify_on=setup_json['data']['stratify_column'],
        )
        
        # Load data split
        dataset.load_data_split(
            split_strategy=setup_json['data']['split_strategy'],
            stratify_on=setup_json['data']['stratify_column'],
        )

    # Dataloaders
    train_loader = DataLoader(dataset.train_set, batch_size=setup_json['data']['batch_size'], shuffle=True, num_workers=setup_json['data']['num_workers'])
    validation_loader = DataLoader(dataset.validation_set, batch_size=setup_json['data']['batch_size'], shuffle=False, num_workers=setup_json['data']['num_workers'])
    
    #%% Prepare training

    # Determine output dimension
    out_dim = 0
    if setup_json['model']['out_dim'] is None:
        for batch in train_loader:
            if setup_json['data']['graph_type'] == 'unit_cell':
                _out_dim = torch.amax(batch.y['n_atoms'])
            else:
                _out_dim = torch.amax(batch.y['unit_cell_n_atoms'])
            if _out_dim > out_dim:
                out_dim = _out_dim
        setup_json['model']['out_dim'] = out_dim.item()
    
    # Calculate normalization factors for cell parameters
    if setup_json['data']['normalize_cell_parameters']:
        cell_a = []
        cell_b = []
        cell_c = []
        cell_alpha = []
        cell_beta = []
        cell_gamma = []
        for batch in train_loader:
            cell_a.extend(batch.y['cell_params'][:,0].tolist())
            cell_b.extend(batch.y['cell_params'][:,1].tolist())
            cell_c.extend(batch.y['cell_params'][:,2].tolist())
            cell_alpha.extend(batch.y['cell_params'][:,3].tolist())
            cell_beta.extend(batch.y['cell_params'][:,4].tolist())
            cell_gamma.extend(batch.y['cell_params'][:,5].tolist())
        
        eps = 1e-10
        
        setup_json['data']['cell_normalization']['a']['mean'] = np.mean(cell_a)
        setup_json['data']['cell_normalization']['a']['std'] = np.std(cell_a) + eps
        setup_json['data']['cell_normalization']['b']['mean'] = np.mean(cell_b)
        setup_json['data']['cell_normalization']['b']['std'] = np.std(cell_b) + eps
        setup_json['data']['cell_normalization']['c']['mean'] = np.mean(cell_c)
        setup_json['data']['cell_normalization']['c']['std'] = np.std(cell_c) + eps
        setup_json['data']['cell_normalization']['alpha']['mean'] = 0 #np.mean(cell_alpha)
        setup_json['data']['cell_normalization']['alpha']['std'] = 180 #np.std(cell_alpha) + eps
        setup_json['data']['cell_normalization']['beta']['mean'] = 0 #np.mean(cell_beta)
        setup_json['data']['cell_normalization']['beta']['std'] = 180 #np.std(cell_beta) + eps
        setup_json['data']['cell_normalization']['gamma']['mean'] = 0 #np.mean(cell_gamma)
        setup_json['data']['cell_normalization']['gamma']['std'] = 180 #np.std(cell_gamma) + eps
    
        # Free up memory
        del cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma
    
    # Calculate normalization factors for atom positions
    if setup_json['data']['normalize_atom_positions']:
        atom_positions = []
        for batch in train_loader:
            atom_positions.extend(batch.pos_abs.tolist())
        
        atom_positions = np.array(atom_positions).flatten()

        eps = 1e-10
        
        setup_json['data']['atom_position_normalization']['mean'] = np.mean(atom_positions, axis=0)
        setup_json['data']['atom_position_normalization']['std'] = np.std(atom_positions, axis=0) + eps
        
        # Free up memory
        del atom_positions

    # Calculate normalization factors for distances
    if setup_json['data']['normalize_distances']:
        distances = []
        for batch in train_loader:
            distances.extend(batch.edge_attr.tolist())
        
        distances = np.array(distances)
        
        eps = 1e-10
        
        setup_json['data']['distance_normalization']['mean'] = np.mean(distances, axis=0)
        setup_json['data']['distance_normalization']['std'] = np.std(distances, axis=0) + eps
        
        # Free up memory
        del distances

    # Save setup json in model directory
    with open(experiment_folder + '/setup_json.json', 'w') as f:
        json.dump(setup_json, f, indent=4)
    
    # Instantiate model
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
    
    # Print model summary
    print(model)
    
    # Load checkpoint if specified
    if setup_json['start_from_checkpoint'] is not None:
        model.load_state_dict(torch.load(setup_json['checkpoint']))
    else:
        # Set checkpoint to last model
        setup_json['start_from_checkpoint'] = f'{experiment_folder}/latest_model.pth'

        # Save setup json in model directory
        with open(experiment_folder + '/setup_json.json', 'w') as f:
            json.dump(setup_json, f, indent=4)
        
    # Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=setup_json['training']['learning_rate'])
    
    # Loss functions
    loss_fn_cell_parameters = torch.nn.MSELoss()
    # loss_fn_cell_positions = torch.nn.MSELoss()
    # loss_fn_cell_atoms = torch.nn.CrossEntropyLoss()
    # loss_fn_kld = torch.nn.KLDivLoss()
    loss_fn_cell_positions = weighted_MSELoss()
    loss_fn_cell_atoms = weighted_CrossEntropyLoss()
    loss_fn_latent_mean = torch.nn.MSELoss()
    loss_fn_latent_std = torch.nn.MSELoss()
        
    #%% Train model

    # Setup logging file
    with open(f'{experiment_folder}/training_log.csv', 'w') as f:
        f.write('epoch,train_loss,train_loss_reconstruction,train_loss_cell_parameters,train_loss_cell_positions,train_loss_cell_atoms,train_loss_kld,validation_loss,validation_loss_reconstruction,validation_loss_cell_parameters,validation_loss_cell_positions,validation_loss_cell_atoms,validation_loss_kld,stage\n')

    # Load normalization factors
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
    best_loss = np.inf
    patience = setup_json['training']['patience']
    patience_counter = 0
    best_epoch = 0
    pretraining = setup_json['training']['seperate_decoder']
    seperate_decoder = setup_json['training']['seperate_decoder']
    
    for epoch in range(setup_json['training']['epochs']):
        # Check patience
        if patience_counter >= patience:
            if seperate_decoder and pretraining:
                pretraining = False
                patience_counter = 0
                best_loss = np.inf
                print('Pretraining done, starting training of main decoder')
            else:
                print(f'Early stopping after {epoch - 1} epochs')
                break
        
        # Train model
        model.train()
        
        if seperate_decoder and not pretraining:
            # Freeze parameters of the encoder and the prior
            # Encoder
            model.scattering_encoder.requires_grad_(False)
            model.graph_encoder_local.requires_grad_(False)
            model.graph_encoder_mlp.requires_grad_(False)
            model.composition_encoder.requires_grad_(False)
            model.linear_encoder.requires_grad_(False)
            
            # Prior
            model.prior_composition_encoder.requires_grad_(False)
            model.prior_scattering_encoder.requires_grad_(False)
            model.prior_linear_encoder.requires_grad_(False)
        
        # Setup for logging loss
        train_loss = 0
        train_loss_rec = 0
        train_loss_cell_parameters = 0
        train_loss_cell_positions = 0
        train_loss_cell_atoms = 0
        train_loss_kld = 0

        # Training loop
        for batch in tqdm(train_loader, desc='Training', leave=False, disable=setup_json['disable_tqdm']):
            
            # Put batch on device
            batch = batch.to(device)
            this_batch_size = batch.batch.amax().item() + 1
            # Zero gradients
            optimizer.zero_grad()
            
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
                composition = composition,
                edge_attr = batch_distances, 
                batch = batch.batch,
                pretraining=pretraining,
            )
            
            if setup_json['data']['graph_type'] in ['unit_cell', 'super_cell']:
                cell_positions_true = batch.pos_frac
                cell_atoms_true = batch.x[:,0]
            elif setup_json['data']['graph_type'] == 'combi':
                if pretraining:
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
                    cell_positions_true = batch.pos_frac_target
                    cell_atoms_true = batch.x_target[:,0]
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
                # Map atom numbers of ligands to logit 1 (Ligand)
                for ligand in setup_json['training']['ligands']:
                    cell_atoms_true = torch.where(cell_atoms_true == ligand, 1, cell_atoms_true)
                # Map all other atom numbers to logit 2 (Metal)
                cell_atoms_true = torch.where(cell_atoms_true >= 2, 2, cell_atoms_true)
            
            # Loss calculation
            loss_cell_parameters = loss_fn_cell_parameters(cell_parameters, cell_parameters_true) 
            
            # loss_cell_positions = loss_fn_cell_positions(cell_positions, cell_positions_true) # Unweighted
            loss_cell_positions = loss_fn_cell_positions(cell_positions, cell_positions_true, cell_positions_weights) # Weighted
            
            # loss_cell_atoms = loss_fn_cell_atoms(cell_atoms, cell_atoms_true) # Unweighted
            loss_cell_atoms = loss_fn_cell_atoms(cell_atoms, cell_atoms_true, cell_atoms_weights) # Weighted
            
            loss_kld = kld.mean()
            
            loss_latent_mean = loss_fn_latent_mean(prior_mean, post_mean)
            loss_latent_std = loss_fn_latent_std(prior_log_std, post_log_std)
            
            reconstruction_loss = loss_cell_parameters + loss_cell_positions + loss_cell_atoms
            
            total_loss = torch.log(reconstruction_loss + ((loss_kld + loss_latent_mean + loss_latent_std) * beta)) #torch.log(reconstruction_loss) + (loss_kld * beta)
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Store loss
            train_loss += total_loss.item()
            train_loss_rec += reconstruction_loss.item()
            train_loss_cell_parameters += loss_cell_parameters.item()
            train_loss_cell_positions += loss_cell_positions.item()
            train_loss_cell_atoms += loss_cell_atoms.item()
            train_loss_kld += loss_kld.item()
            
            
        # Calculate average loss
        train_loss /= len(train_loader)
        train_loss_rec /= len(train_loader)
        train_loss_cell_parameters /= len(train_loader)
        train_loss_cell_positions /= len(train_loader)
        train_loss_cell_atoms /= len(train_loader)
        train_loss_kld /= len(train_loader)
        
        # Validate model
        model.eval()

        # Setup for logging loss
        validation_loss = 0
        validation_loss_rec = 0
        validation_loss_cell_parameters = 0
        validation_loss_cell_positions = 0
        validation_loss_cell_atoms = 0
        validation_loss_kld = 0

        # Validation loop
        with torch.no_grad():
            for batch in tqdm(validation_loader, desc='Validation', leave=False, disable=setup_json['disable_tqdm']):
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
                    composition = composition,
                    edge_attr = batch_distances, 
                    batch = batch.batch,
                    pretraining=pretraining,
                )
                
                if setup_json['data']['graph_type'] in ['unit_cell', 'super_cell']:
                    cell_positions_true = batch.pos_frac
                    cell_atoms_true = batch.x[:,0]
                elif setup_json['data']['graph_type'] == 'combi':
                    if pretraining:
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
                        cell_positions_true = batch.pos_frac_target
                        cell_atoms_true = batch.x_target[:,0]
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
                
                # Free up memory related to batch
                del batch, batch_scattering, batch_positions, batch_distances, batch_features
                
                # Reshape atom predictions
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
                    # Map atom numbers of ligands to logit 1 (Ligand)
                    for ligand in setup_json['training']['ligands']:
                        cell_atoms_true = torch.where(cell_atoms_true == ligand, 1, cell_atoms_true)
                    # Map all other atom numbers to logit 2 (Metal)
                    cell_atoms_true = torch.where(cell_atoms_true >= 2, 2, cell_atoms_true)
                
                # Rounding positions to 5 decimals
                cell_positions = torch.round(cell_positions, decimals=5)
                
                # Loss
                loss_cell_parameters = loss_fn_cell_parameters(cell_parameters, cell_parameters_true) 
                
                # loss_cell_positions = loss_fn_cell_positions(cell_positions, cell_positions_true) # Unweighted
                loss_cell_positions = loss_fn_cell_positions(cell_positions, cell_positions_true, cell_positions_weights) # Weighted
                
                # loss_cell_atoms = loss_fn_cell_atoms(cell_atoms, cell_atoms_true) # Unweighted
                loss_cell_atoms = loss_fn_cell_atoms(cell_atoms, cell_atoms_true, cell_atoms_weights) # Weighted
                
                loss_kld = kld.mean()
                
                loss_latent_mean = loss_fn_latent_mean(prior_mean, post_mean)
                loss_latent_std = loss_fn_latent_std(prior_log_std, post_log_std)
                
                reconstruction_loss = loss_cell_parameters + loss_cell_positions + loss_cell_atoms
                
                total_loss = torch.log(reconstruction_loss + ((loss_kld + loss_latent_mean + loss_latent_std) * beta)) #torch.log(reconstruction_loss) + (loss_kld * beta)
                
                # Store loss
                validation_loss += total_loss.item()
                validation_loss_rec += reconstruction_loss.item()
                validation_loss_cell_parameters += loss_cell_parameters.item()
                validation_loss_cell_positions += loss_cell_positions.item()
                validation_loss_cell_atoms += loss_cell_atoms.item()
                validation_loss_kld += loss_kld.item()

        # Calculate average loss
        validation_loss /= len(validation_loader)
        validation_loss_rec /= len(validation_loader)
        validation_loss_cell_parameters /= len(validation_loader)
        validation_loss_cell_positions /= len(validation_loader)
        validation_loss_cell_atoms /= len(validation_loader)
        validation_loss_kld /= len(validation_loader)
        
        # Check if model improved
        if validation_loss_rec < best_loss:
            patience_counter = 0
            best_epoch = epoch
            best_loss = validation_loss_rec
            torch.save(model.state_dict(), f'{experiment_folder}/best_model.pth')
        else:
            patience_counter += 1
            
        # Save latest model
        torch.save(model.state_dict(), f'{experiment_folder}/latest_model.pth')
        
        # Save loss in log file
        with open(f'{experiment_folder}/training_log.csv', 'a') as f:
            stage = 'Pretraining' if pretraining else 'Training'
            f.write(f'{epoch},{train_loss},{train_loss_rec},{train_loss_cell_parameters},{train_loss_cell_positions},{train_loss_cell_atoms},{train_loss_kld},{validation_loss},{validation_loss_rec},{validation_loss_cell_parameters},{validation_loss_cell_positions},{validation_loss_cell_atoms},{validation_loss_kld},{stage}\n')


        # Print progress
        print(f'Epoch: {epoch} | Train loss: {train_loss:.2e} | Validation loss: {validation_loss:.2e} | Best reconstruction loss: {best_loss:.2e} (Epoch {best_epoch}) | Patience: {patience_counter}/{patience}')

    # Record date and time
    setup_json['experiment_end'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Set checkpoint to best model
    setup_json['start_from_checkpoint'] = f'{experiment_folder}/best_model.pth'
    
    # Save setup json in model directory
    with open(experiment_folder + '/setup_json.json', 'w') as f:
        json.dump(setup_json, f, indent=4)