#%% Imports
import argparse
import json
import warnings
import numpy as np
from modules.CHILI import CHILI
from modules.net import SCVAE
from torch_geometric.loader import DataLoader
from torch_geometric.nn import Sequential as pyg_Sequential
import torch
import datetime
import pathlib
from tqdm.auto import tqdm

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
    experiment_folder = f'{setup_json["model_root"]}/{setup_json["experiment_name"]}'
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
            _out_dim = torch.amax(batch.y['unit_cell_n_atoms'])
            if _out_dim > out_dim:
                out_dim = _out_dim
        setup_json['model']['out_dim'] = out_dim.item()
        
    # Save setup json in model directory
    with open(experiment_folder + '/setup_json.json', 'w') as f:
        json.dump(setup_json, f, indent=4)
    
    # Instantiate model
    model = SCVAE(
        latent_dim=setup_json['model']['latent_dim'],
        out_dim=setup_json['model']['out_dim'],
    )
    model = model.to(device)
    
    # Print model summary
    print(model)
    
    # Load checkpoint if specified
    if setup_json['start_from_checkpoint'] is not None:
        model.load_state_dict(torch.load(setup_json['checkpoint']))
        
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=setup_json['training']['learning_rate'])
    
    # Loss functions
    loss_fn_cell_parameters = torch.nn.MSELoss()
    loss_fn_cell_positions = torch.nn.MSELoss()
    loss_fn_cell_atoms = torch.nn.CrossEntropyLoss()
    loss_fn_kld = torch.nn.KLDivLoss()
        
    #%% Train model
    best_loss = np.inf
    patience = setup_json['training']['patience']
    patience_counter = 0
    best_epoch = 0
    for epoch in range(setup_json['training']['epochs']):
        # Check patience
        if patience_counter >= patience:
            print(f'Early stopping after {epoch} epochs')
            break
        # Train model
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc='Training', leave=False):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            batch = batch.to(device)
            # print(batch.is_cuda)
            # print(next(model.parameters()).is_cuda)
            cell_parameters, cell_positions, cell_atoms, mean, log_std = model.forward(
                x = torch.cat((batch.x, batch.pos_abs), dim=1), 
                edge_index = batch.edge_index, 
                edge_attr = batch.edge_attr, 
                batch = batch.batch,
            )

            # Loss
            loss_cell_parameters = loss_fn_cell_parameters(cell_parameters, batch.y['cell_params'].view(-1, 6))
            loss_cell_positions = 0 # Placeholder
            loss_cell_atoms = 0 # Placeholder
            loss_kld = 0 # Placeholder
            
            total_loss = loss_cell_parameters + loss_cell_positions + loss_cell_atoms + loss_kld
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Store loss
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate model
        model.eval()
        validation_loss = 0
        for batch in tqdm(validation_loader, desc='Validation', leave=False):
            # Forward pass
            batch = batch.to(device)
            cell_parameters, cell_positions, cell_atoms, mean, log_std = model.forward(
                x = torch.cat((batch.x, batch.pos_abs), dim=1), 
                edge_index = batch.edge_index, 
                edge_attr = batch.edge_attr, 
                batch = batch.batch,
            )

            # Loss
            loss_cell_parameters = loss_fn_cell_parameters(cell_parameters, batch.y['cell_params'].view(-1, 6))
            loss_cell_positions = 0 # Placeholder
            loss_cell_atoms = 0 # Placeholder
            loss_kld = 0 # Placeholder
            
            total_loss = loss_cell_parameters + loss_cell_positions + loss_cell_atoms + loss_kld
            
            # Store loss
            validation_loss += total_loss.item()

        validation_loss /= len(validation_loader)
        
        # Check if model improved
        if validation_loss < best_loss:
            patience_counter = 0
            best_epoch = epoch
            best_loss = validation_loss
            torch.save(model.state_dict(), f'{experiment_folder}/best_model.pth')
        else:
            patience_counter += 1
            
        # Save latest model
        torch.save(model.state_dict(), f'{experiment_folder}/latest_model.pth')
        
        # Print progress
        print(f'Epoch: {epoch} | Train loss: {train_loss:.2e} | Validation loss: {validation_loss:.2e} | Best loss: {best_loss:.2e} (Epoch {best_epoch}) | Patience: {patience_counter}/{patience}')
