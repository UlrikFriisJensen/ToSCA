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
    train_loader = DataLoader(dataset.train_set, batch_size=setup_json['data']['batch_size'], shuffle=True)
    validation_loader = DataLoader(dataset.validation_set, batch_size=setup_json['data']['batch_size'], shuffle=False)
    
    #%% Prepare training

    # Instantiate model
    model = SCVAE().to(device)
        
    # Load checkpoint if specified
    if setup_json['start_from_checkpoint'] is not None:
        model.load_state_dict(torch.load(setup_json['checkpoint']))
        
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=setup_json['training']['learning_rate'])
    
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
        
    #%% Train model

    for epoch in range(setup_json['training']['epochs']):
        # Train model
        for batch in train_loader:
            batch = batch.to(device)
            
            
            raise ValueError
            pass

        # Validate model
        pass

        # Save model
        pass
