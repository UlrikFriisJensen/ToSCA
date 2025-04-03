#%% Imports

import argparse
import numpy as np
import pandas as pd
from ase import Atoms
from pathlib import Path
from ase.io import read, write
from ase.build import make_supercell
from ase.build import sort as ase_sort
from tqdm.auto import tqdm

#%% Functions

def rocksalt_transformation(rocksalt_unitcell, expansion_factor=2, translation=0.25, inversion=False):
    # Make a supercell of the unit cell
    rocksalt_supercell = make_supercell(rocksalt_unitcell, np.diag([expansion_factor, expansion_factor, expansion_factor]))

    if inversion:
        # Invert the atoms in the supercell
        rocksalt_supercell.set_scaled_positions(1 - rocksalt_supercell.get_scaled_positions())

    # Translate the atoms in the supercell
    rocksalt_supercell.set_scaled_positions(rocksalt_supercell.get_scaled_positions() + translation) 
    
    return rocksalt_supercell

def spinel_transformation(spinel_unitcell, expansion_factor=1, translation=0.0, inversion=False):
    # Make a supercell of the unit cell
    spinel_supercell = make_supercell(spinel_unitcell, np.diag([expansion_factor, expansion_factor, expansion_factor]))
    
    if inversion:
        # Invert the atoms in the supercell
        spinel_supercell.set_scaled_positions(1 - spinel_supercell.get_scaled_positions())

    # Translate the atoms in the supercell
    spinel_supercell.set_scaled_positions(spinel_supercell.get_scaled_positions() + translation) 
    
    return spinel_supercell

def zincblende_transformation(zincblende_unitcell, expansion_factor=2,  translation=0.0, inversion=True):
    # Make a supercell of the unit cell
    zincblende_supercell = make_supercell(zincblende_unitcell, np.diag([expansion_factor, expansion_factor, expansion_factor]))
    
    if inversion:
        # Invert the atoms in the supercell
        zincblende_supercell.set_scaled_positions(1 - zincblende_supercell.get_scaled_positions())

    # Translate the atoms in the supercell
    zincblende_supercell.set_scaled_positions(zincblende_supercell.get_scaled_positions() + translation) 
    
    return zincblende_supercell

def nickelArsenide_transformation(nickelArsenide_unitcell, expansion_factor=2, translation=0.0, inversion=False):
    # Make a supercell of the unit cell
    nickelArsenide_supercell = make_supercell(nickelArsenide_unitcell, np.diag([expansion_factor, expansion_factor, expansion_factor]))

    if inversion:
        # Invert the atoms in the supercell
        nickelArsenide_supercell.set_scaled_positions(1 - nickelArsenide_supercell.get_scaled_positions())

    # Translate the atoms in the supercell
    nickelArsenide_supercell.set_scaled_positions(nickelArsenide_supercell.get_scaled_positions() + translation) 
    
    return nickelArsenide_supercell

def cadmiumIodide_transformation(cadmiumIodide_unitcell, expansion_factor=2, translation=0.0, inversion=False):
    # Make a supercell of the unit cell
    cadmiumIodide_supercell = make_supercell(cadmiumIodide_unitcell, np.diag([expansion_factor, expansion_factor, expansion_factor]))
    
    if inversion:
        # Invert the atoms in the supercell
        cadmiumIodide_supercell.set_scaled_positions(1 - cadmiumIodide_supercell.get_scaled_positions())
        
    # Translate the atoms in the supercell
    cadmiumIodide_supercell.set_scaled_positions(cadmiumIodide_supercell.get_scaled_positions() + translation)
    
    return cadmiumIodide_supercell

def sigmoid_function(x, center=0.5, steepness=10):
    return 1 / (1 + np.exp(-steepness * (x - center)))

def rocksalt_to_spinel(rocksalt_structure, spinel_structure, interpolation_steps=10, atom_samples=5, identification_threshold=0.1):
    # Sort the atoms
    rocksalt_structure = ase_sort(rocksalt_structure, tags=rocksalt_structure.get_atomic_numbers())
    spinel_structure = ase_sort(spinel_structure, tags=spinel_structure.get_atomic_numbers())    
    
    # Get cell parameters
    rocksalt_cell_parameters = rocksalt_structure.cell.cellpar()
    spinel_cell_parameters = spinel_structure.cell.cellpar()
    
    # Get the scaled positions of the atoms
    rocksalt_positions = rocksalt_structure.get_scaled_positions()
    spinel_positions = spinel_structure.get_scaled_positions()
    
    # Get the atomic numbers of the atoms
    rocksalt_atoms = rocksalt_structure.get_atomic_numbers()
    spinel_atoms = spinel_structure.get_atomic_numbers()
    
    # Find the positions of the oxygen atoms
    rocksalt_O_positions = rocksalt_positions[rocksalt_atoms == 8]
    spinel_O_positions = spinel_positions[spinel_atoms == 8]
    
    # Sort the oxygen atoms by distance to the origo and then by coordinate TODO: Do this
    temp_O_df = pd.DataFrame(rocksalt_O_positions, columns=['x', 'y', 'z'])
    temp_O_df['distance'] = np.linalg.norm(rocksalt_O_positions, axis=1)
    rocksalt_O_positions = rocksalt_O_positions[temp_O_df.sort_values(['distance', 'x', 'y', 'z']).index]
    
    temp_O_df = pd.DataFrame(spinel_O_positions, columns=['x', 'y', 'z'])
    temp_O_df['distance'] = np.linalg.norm(spinel_O_positions, axis=1)
    spinel_O_positions = spinel_O_positions[temp_O_df.sort_values(['distance', 'x', 'y', 'z']).index]

    del temp_O_df
    
    # Find the positions of the metal atoms
    rocksalt_M_positions = rocksalt_positions[rocksalt_atoms != 8]
    spinel_M_positions = spinel_positions[spinel_atoms != 8]
    
    # Structure metallic element
    metallic_element = rocksalt_structure.get_atomic_numbers()[-1]
    
    # Find the metal atoms that are in both the rocksalt structure and the spinel structure
    rocksalt_common_M_positions = []
    spinel_common_M_positions = []
    for index in range(len(rocksalt_M_positions)):
        distances = np.linalg.norm(spinel_M_positions - rocksalt_M_positions[index], axis=1)
        if np.min(distances) < identification_threshold:
            rocksalt_common_M_positions.append(index)
            spinel_common_M_positions.append(np.argmin(distances))
            
    rocksalt_different_M_positions = [index for index in range(len(rocksalt_M_positions)) if index not in rocksalt_common_M_positions]
    spinel_different_M_positions = [index for index in range(len(spinel_M_positions)) if index not in spinel_common_M_positions]
    
    interp_step_list = []
    sample_i_list = []
    interpolated_cell_parameters_list = []
    interpolated_atoms_list = []
    interpolated_positions_list = []
    for interp_step_i in range(interpolation_steps + 1):
        if interp_step_i == 0:
            # Save the rocksalt structure
            interp_step_list.append(interp_step_i)
            sample_i_list.append(0)
            interpolated_cell_parameters_list.append(rocksalt_cell_parameters)
            interpolated_atoms_list.append(rocksalt_atoms)
            interpolated_positions_list.append(rocksalt_positions)
            continue
        elif interp_step_i == interpolation_steps:
            # Save the spinel structure
            interp_step_list.append(interp_step_i)
            sample_i_list.append(0)
            interpolated_cell_parameters_list.append(spinel_cell_parameters)
            interpolated_atoms_list.append(spinel_atoms)
            interpolated_positions_list.append(spinel_positions)
            continue
        else:
            # Interpolate the cell parameters 
            interpolated_cell_parameters = rocksalt_cell_parameters + (spinel_cell_parameters - rocksalt_cell_parameters) * interp_step_i / interpolation_steps
            
            # Interpolate the positions of oxygens
            interpolated_O_positions = rocksalt_O_positions + (spinel_O_positions - rocksalt_O_positions) * interp_step_i / interpolation_steps
            # interpolated_O_list.append(interpolated_O_positions)
            
            # Interpolate the positions of the common metal atoms
            interpolated_common_M_positions = rocksalt_M_positions[rocksalt_common_M_positions] + (spinel_M_positions[spinel_common_M_positions] - rocksalt_M_positions[rocksalt_common_M_positions]) * interp_step_i / interpolation_steps
            # interpolated_common_M_list.append(interpolated_common_M_positions)
            
            for sample_i in range(atom_samples):                            
                # Non-shared metal atoms from rock salt to keep
                n_rocksalt_different_M_positions = len(rocksalt_different_M_positions)
                
                # sigmoid probability for keeping an atom
                p_1 = 1 - sigmoid_function(interp_step_i / interpolation_steps, center=0.4, steepness=10)
                
                rocksalt_indeces_to_keep = np.where(np.random.random(n_rocksalt_different_M_positions) < p_1)[0]
                
                interpolated_different_M_positions = rocksalt_M_positions[rocksalt_different_M_positions][rocksalt_indeces_to_keep]
            
                # Non-shared metal atoms from spinel to add
                # sigmoid probability for adding the atom
                p_2 = sigmoid_function(interp_step_i / interpolation_steps, center=0.6, steepness=10)
                
                spinel_indeces_to_add = np.where(np.random.random(len(spinel_different_M_positions)) < p_2)[0]
                
                # Check that there are not more atoms to add than there are atoms to remove
                if len(spinel_indeces_to_add) > n_rocksalt_different_M_positions - len(rocksalt_indeces_to_keep):
                    spinel_indeces_to_add = np.random.choice(spinel_indeces_to_add, n_rocksalt_different_M_positions - len(rocksalt_indeces_to_keep), replace=False)
                
                interpolated_different_M_positions = np.concatenate([interpolated_different_M_positions, spinel_M_positions[spinel_different_M_positions][spinel_indeces_to_add]])
                
                # Construct the interpolated structure
                interpolated_atoms = np.concatenate(
                    [
                        [8] * len(interpolated_O_positions),
                        [metallic_element] * len(interpolated_common_M_positions),
                        [metallic_element] * len(interpolated_different_M_positions)
                        ]
                    )
                interpolated_positions = np.concatenate(
                    [
                        interpolated_O_positions, 
                        interpolated_common_M_positions, 
                        interpolated_different_M_positions, 
                    ]
                )
                
                # Save the interpolated structure
                interp_step_list.append(interp_step_i)
                sample_i_list.append(sample_i)
                interpolated_cell_parameters_list.append(interpolated_cell_parameters)
                interpolated_atoms_list.append(interpolated_atoms)
                interpolated_positions_list.append(interpolated_positions)  
        
    return interp_step_list, sample_i_list, interpolated_cell_parameters_list, interpolated_atoms_list, interpolated_positions_list

def zincblende_to_spinel(zincblende_structure, spinel_structure, interpolation_steps=10, atom_samples=5, identification_threshold=0.1):
    # Sort the atoms
    zincblende_structure = ase_sort(zincblende_structure, tags=zincblende_structure.get_atomic_numbers())
    spinel_structure = ase_sort(spinel_structure, tags=spinel_structure.get_atomic_numbers())    
    
    # Get cell parameters
    zincblende_cell_parameters = zincblende_structure.cell.cellpar()
    spinel_cell_parameters = spinel_structure.cell.cellpar()
    
    # Get the scaled positions of the atoms
    zincblende_positions = zincblende_structure.get_scaled_positions()
    spinel_positions = spinel_structure.get_scaled_positions()
    
    # Get the atomic numbers of the atoms
    zincblende_atoms = zincblende_structure.get_atomic_numbers()
    spinel_atoms = spinel_structure.get_atomic_numbers()
    
    # Find the positions of the oxygen atoms
    zincblende_O_positions = zincblende_positions[zincblende_atoms == 8]
    spinel_O_positions = spinel_positions[spinel_atoms == 8]
    
    # Sort the oxygen atoms by distance to the origo and then by coordinate TODO: Do this
    temp_O_df = pd.DataFrame(zincblende_O_positions, columns=['x', 'y', 'z'])
    temp_O_df['distance'] = np.linalg.norm(zincblende_O_positions, axis=1)
    zincblende_O_positions = zincblende_O_positions[temp_O_df.sort_values(['distance', 'x', 'y', 'z']).index]
    
    temp_O_df = pd.DataFrame(spinel_O_positions, columns=['x', 'y', 'z'])
    temp_O_df['distance'] = np.linalg.norm(spinel_O_positions, axis=1)
    spinel_O_positions = spinel_O_positions[temp_O_df.sort_values(['distance', 'x', 'y', 'z']).index]

    del temp_O_df
    
    # Find the positions of the metal atoms
    zincblende_M_positions = zincblende_positions[zincblende_atoms != 8]
    spinel_M_positions = spinel_positions[spinel_atoms != 8]
    
    # Structure metallic element
    metallic_element = zincblende_structure.get_atomic_numbers()[-1]
    
    # Find the metal atoms that are in both the zincblende structure and the spinel structure
    zincblende_common_M_positions = []
    spinel_common_M_positions = []
    for index in range(len(zincblende_M_positions)):
        distances = np.linalg.norm(spinel_M_positions - zincblende_M_positions[index], axis=1)
        if np.min(distances) < identification_threshold:
            zincblende_common_M_positions.append(index)
            spinel_common_M_positions.append(np.argmin(distances))
            
    zincblende_different_M_positions = [index for index in range(len(zincblende_M_positions)) if index not in zincblende_common_M_positions]
    spinel_different_M_positions = [index for index in range(len(spinel_M_positions)) if index not in spinel_common_M_positions]
    
    interp_step_list = []
    sample_i_list = []
    interpolated_cell_parameters_list = []
    interpolated_atoms_list = []
    interpolated_positions_list = []
    
    for interp_step_i in range(interpolation_steps + 1):
        if interp_step_i == 0:
            # Save the zincblende structure
            interp_step_list.append(interp_step_i)
            sample_i_list.append(0)
            interpolated_cell_parameters_list.append(zincblende_cell_parameters)
            interpolated_atoms_list.append(zincblende_atoms)
            interpolated_positions_list.append(zincblende_positions)
            continue
        elif interp_step_i == interpolation_steps:
            # Save the spinel structure
            interp_step_list.append(interp_step_i)
            sample_i_list.append(0)
            interpolated_cell_parameters_list.append(spinel_cell_parameters)
            interpolated_atoms_list.append(spinel_atoms)
            interpolated_positions_list.append(spinel_positions)
            continue
        else:
            # Interpolate the cell parameters 
            interpolated_cell_parameters = zincblende_cell_parameters + (spinel_cell_parameters - zincblende_cell_parameters) * interp_step_i / interpolation_steps
            
            # Interpolate the positions of oxygens
            interpolated_O_positions = zincblende_O_positions + (spinel_O_positions - zincblende_O_positions) * interp_step_i / interpolation_steps
            # interpolated_O_list.append(interpolated_O_positions)
            
            # Interpolate the positions of the common metal atoms
            interpolated_common_M_positions = zincblende_M_positions[zincblende_common_M_positions] + (spinel_M_positions[spinel_common_M_positions] - zincblende_M_positions[zincblende_common_M_positions]) * interp_step_i / interpolation_steps
            # interpolated_common_M_list.append(interpolated_common_M_positions)
            
            for sample_i in range(atom_samples):                           
                # Non-shared metal atoms from zincblende to keep
                n_zincblende_different_M_positions = len(zincblende_different_M_positions)
                
                p_1 = 1 - sigmoid_function(interp_step_i / interpolation_steps, center=0.4, steepness=12)
                
                zincblende_indeces_to_keep = np.where(np.random.random(n_zincblende_different_M_positions) < p_1)[0]
                
                interpolated_different_M_positions = zincblende_M_positions[zincblende_different_M_positions][zincblende_indeces_to_keep]
            
                # Non-shared metal atoms from spinel to add
                p_2 = sigmoid_function(interp_step_i / interpolation_steps, center=0.6, steepness=12)
                
                spinel_indeces_to_add = np.where(np.random.random(len(spinel_different_M_positions)) < p_2)[0]
                
                # Check that there are not more atoms to add than there are atoms to remove
                if len(spinel_indeces_to_add) > n_zincblende_different_M_positions - len(zincblende_indeces_to_keep):
                    spinel_indeces_to_add = np.random.choice(spinel_indeces_to_add, n_zincblende_different_M_positions - len(zincblende_indeces_to_keep), replace=False)
                
                interpolated_different_M_positions = np.concatenate([interpolated_different_M_positions, spinel_M_positions[spinel_different_M_positions][spinel_indeces_to_add]])
                
                # Construct the interpolated structure
                interpolated_atoms = np.concatenate(
                    [
                        [8] * len(interpolated_O_positions),
                        [metallic_element] * len(interpolated_common_M_positions),
                        [metallic_element] * len(interpolated_different_M_positions)
                        ]
                    )
                interpolated_positions = np.concatenate(
                    [
                        interpolated_O_positions, 
                        interpolated_common_M_positions, 
                        interpolated_different_M_positions, 
                    ]
                )
                
                # Save the interpolated structure
                interp_step_list.append(interp_step_i)
                sample_i_list.append(sample_i)
                interpolated_cell_parameters_list.append(interpolated_cell_parameters)
                interpolated_atoms_list.append(interpolated_atoms)
                interpolated_positions_list.append(interpolated_positions)
    
    return interp_step_list, sample_i_list, interpolated_cell_parameters_list, interpolated_atoms_list, interpolated_positions_list

def nickelArsenide_to_cadmiumIodide(nickelArsenide_structure, cadmiumIodide_structure, interpolation_steps=5, atom_samples=5, identification_threshold=0.1):
    # Sort the atoms
    nickelArsenide_structure = ase_sort(nickelArsenide_structure, tags=nickelArsenide_structure.get_atomic_numbers())
    cadmiumIodide_structure = ase_sort(cadmiumIodide_structure, tags=cadmiumIodide_structure.get_atomic_numbers())    
    
    # Get cell parameters
    nickelArsenide_cell_parameters = nickelArsenide_structure.cell.cellpar()
    cadmiumIodide_cell_parameters = cadmiumIodide_structure.cell.cellpar()
    
    # Get the scaled positions of the atoms
    nickelArsenide_positions = nickelArsenide_structure.get_scaled_positions()
    cadmiumIodide_positions = cadmiumIodide_structure.get_scaled_positions()
    
    # Get the atomic numbers of the atoms
    nickelArsenide_atoms = nickelArsenide_structure.get_atomic_numbers()
    cadmiumIodide_atoms = cadmiumIodide_structure.get_atomic_numbers()
    
    # Find the positions of the oxygen atoms
    nickelArsenide_O_positions = nickelArsenide_positions[nickelArsenide_atoms == 8]
    cadmiumIodide_O_positions = cadmiumIodide_positions[cadmiumIodide_atoms == 8]
    
    # Sort the oxygen atoms by distance to the origo and then by coordinate TODO: Do this
    temp_O_df = pd.DataFrame(nickelArsenide_O_positions, columns=['x', 'y', 'z'])
    temp_O_df['distance'] = np.linalg.norm(nickelArsenide_O_positions, axis=1)
    nickelArsenide_O_positions = nickelArsenide_O_positions[temp_O_df.sort_values(['distance', 'x', 'y', 'z']).index]
    
    temp_O_df = pd.DataFrame(cadmiumIodide_O_positions, columns=['x', 'y', 'z'])
    temp_O_df['distance'] = np.linalg.norm(cadmiumIodide_O_positions, axis=1)
    cadmiumIodide_O_positions = cadmiumIodide_O_positions[temp_O_df.sort_values(['distance', 'x', 'y', 'z']).index]

    del temp_O_df
    
    # Find the positions of the metal atoms
    nickelArsenide_M_positions = nickelArsenide_positions[nickelArsenide_atoms != 8]
    cadmiumIodide_M_positions = cadmiumIodide_positions[cadmiumIodide_atoms != 8]
    
    # Structure metallic element
    metallic_element = nickelArsenide_structure.get_atomic_numbers()[-1]
    
    # Find the metal atoms that are in both the nickelArsenide structure and the cadmiumIodide structure
    nickelArsenide_common_M_positions = []
    cadmiumIodide_common_M_positions = []
    for index in range(len(nickelArsenide_M_positions)):
        distances = np.linalg.norm(cadmiumIodide_M_positions - nickelArsenide_M_positions[index], axis=1)
        if np.min(distances) < identification_threshold:
            nickelArsenide_common_M_positions.append(index)
            cadmiumIodide_common_M_positions.append(np.argmin(distances))
    
    nickelArsenide_different_M_positions = [index for index in range(len(nickelArsenide_M_positions)) if index not in nickelArsenide_common_M_positions]
    cadmiumIodide_different_M_positions = [index for index in range(len(cadmiumIodide_M_positions)) if index not in cadmiumIodide_common_M_positions]
    
    interp_step_list = []
    sample_i_list = []
    interpolated_cell_parameters_list = []
    interpolated_atoms_list = []
    interpolated_positions_list = []
    for interp_step_i in range(interpolation_steps + 1):
        if interp_step_i == 0:
            # Save the nickelArsenide structure
            interp_step_list.append(interp_step_i)
            sample_i_list.append(0)
            interpolated_cell_parameters_list.append(nickelArsenide_cell_parameters)
            interpolated_atoms_list.append(nickelArsenide_atoms)
            interpolated_positions_list.append(nickelArsenide_positions)
            continue
        elif interp_step_i == interpolation_steps:
            # Save the cadmiumIodide structure
            interp_step_list.append(interp_step_i)
            sample_i_list.append(0)
            interpolated_cell_parameters_list.append(cadmiumIodide_cell_parameters)
            interpolated_atoms_list.append(cadmiumIodide_atoms)
            interpolated_positions_list.append(cadmiumIodide_positions)
            continue
        else:
            # Interpolate the cell parameters 
            interpolated_cell_parameters = nickelArsenide_cell_parameters + ((cadmiumIodide_cell_parameters - nickelArsenide_cell_parameters) * interp_step_i / interpolation_steps)
            
            # Interpolate the positions of oxygens
            interpolated_O_positions = nickelArsenide_O_positions + ((cadmiumIodide_O_positions - nickelArsenide_O_positions) * interp_step_i / interpolation_steps)
            
            # Interpolate the positions of the common metal atoms
            interpolated_common_M_positions = nickelArsenide_M_positions[nickelArsenide_common_M_positions] + ((cadmiumIodide_M_positions[cadmiumIodide_common_M_positions] - nickelArsenide_M_positions[nickelArsenide_common_M_positions]) * interp_step_i / interpolation_steps)
            
            for sample_i in range(atom_samples):                            
                # Non-shared metal atoms from nickel arsenide to keep
                n_nickelArsenide_different_M_positions = len(nickelArsenide_different_M_positions)
                
                # sigmoid probability for keeping an atom
                p_1 = 1 - (interp_step_i / interpolation_steps)
                
                nickelArsenide_indeces_to_keep = np.where(np.random.random(n_nickelArsenide_different_M_positions) < p_1)[0]

                interpolated_different_M_positions = nickelArsenide_M_positions[nickelArsenide_different_M_positions][nickelArsenide_indeces_to_keep]
                
                # Construct the interpolated structure
                interpolated_atoms = np.concatenate(
                    [
                        [8] * len(interpolated_O_positions),
                        [metallic_element] * len(interpolated_common_M_positions),
                        [metallic_element] * len(interpolated_different_M_positions)
                        ]
                    )
                interpolated_positions = np.concatenate(
                    [
                        interpolated_O_positions, 
                        interpolated_common_M_positions, 
                        interpolated_different_M_positions, 
                    ]
                )
                
                # Save the interpolated structure
                interp_step_list.append(interp_step_i)
                sample_i_list.append(sample_i)
                interpolated_cell_parameters_list.append(interpolated_cell_parameters)
                interpolated_atoms_list.append(interpolated_atoms)
                interpolated_positions_list.append(interpolated_positions)  
        
    return interp_step_list, sample_i_list, interpolated_cell_parameters_list, interpolated_atoms_list, interpolated_positions_list

#%% Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
    parser.add_argument('--interpolation_type', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--interpolation_steps', type=int, default=10)
    parser.add_argument('--atom_samples', type=int, default=5)
    parser.add_argument('--identification_threshold', type=float, default=0.1)
    
    args = parser.parse_args()
    
    # Make the output folder
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)
    
    # Seed for reproducibility
    np.random.seed(args.seed)
    
    if args.interpolation_type == 'rocksalt_to_spinel_to_zincblende':
        # Get the relevant CIF files
        
        # Rock Salt
        rocksalt_structures = [str(filepath) for filepath in Path(args.cif_folder).rglob('RockSalt*.cif')]
        rocksalt_compositions = [filepath.split('/')[-1].split('_')[-1].split('.')[0] for filepath in rocksalt_structures]
        rocksalt_metals = [composition[:-1] for composition in rocksalt_compositions]
        
        # Spinel
        spinel_structures = [str(filepath) for filepath in Path(args.cif_folder).rglob('Spinel*.cif')]
        spinel_compositions = [filepath.split('/')[-1].split('_')[-1].split('.')[0] for filepath in spinel_structures]
        spinel_metals = [composition[: len(composition[:-3]) // 2] for composition in spinel_compositions]
        
        # Zinc Blende
        zincblende_structures = [str(filepath) for filepath in Path(args.cif_folder).rglob('ZincBlende*.cif')]
        zincblende_compositions = [filepath.split('/')[-1].split('_')[-1].split('.')[0] for filepath in zincblende_structures]
        zincblende_metals = [composition[:-1] for composition in zincblende_compositions]
        
        # Generate the interpolated structures
        for metal in tqdm(rocksalt_metals, desc='Generating interpolated structures'):
            # Read the structures
            # Rock Salt
            rocksalt_structure = read(rocksalt_structures[rocksalt_metals.index(metal)])
            rocksalt_transformed = rocksalt_transformation(rocksalt_structure, expansion_factor=2, translation=0.25, inversion=False)
            
            # Spinel
            spinel_structure = read(spinel_structures[spinel_metals.index(metal)])
            spinel_transformed = spinel_transformation(spinel_structure, expansion_factor=1, translation=0.1374, inversion=False)
            
            # Zinc Blende
            zincblende_structure = read(zincblende_structures[zincblende_metals.index(metal)])
            zincblende_transformed = zincblende_transformation(zincblende_structure, expansion_factor=2, translation=0.125, inversion=True)
            
            # Generate rocksalt to spinel interpolations
            interp_step_list, sample_i_list, interpolated_cell_parameters_list, interpolated_atoms_list, interpolated_positions_list = rocksalt_to_spinel(rocksalt_transformed, spinel_transformed, interpolation_steps=args.interpolation_steps, atom_samples=args.atom_samples, identification_threshold=args.identification_threshold)
            
            # Check for and remove duplicate structures
            duplicate_indices = []
            for i in range(len(interp_step_list)):
                for j in range(i, len(interp_step_list)):
                    if i == j:
                        continue
                    if np.array_equal(interpolated_positions_list[i], interpolated_positions_list[j]) and np.array_equal(interpolated_cell_parameters_list[i], interpolated_cell_parameters_list[j]):
                        if j == len(interp_step_list) - 1:
                            duplicate_indices.append(i)
                        else:
                            duplicate_indices.append(j)

            interp_step_list = [interp_step_list[i] for i in range(len(interp_step_list)) if i not in duplicate_indices]
            sample_i_list = [sample_i_list[i] for i in range(len(sample_i_list)) if i not in duplicate_indices]
            interpolated_cell_parameters_list = [interpolated_cell_parameters_list[i] for i in range(len(interpolated_cell_parameters_list)) if i not in duplicate_indices]
            interpolated_atoms_list = [interpolated_atoms_list[i] for i in range(len(interpolated_atoms_list)) if i not in duplicate_indices]
            interpolated_positions_list = [interpolated_positions_list[i] for i in range(len(interpolated_positions_list)) if i not in duplicate_indices]
            
            # Save interpolations as cif files
            for interpolation_i in range(len(interp_step_list)):
                interpolated_structure = Atoms(
                    cell=interpolated_cell_parameters_list[interpolation_i],
                    scaled_positions=interpolated_positions_list[interpolation_i],
                    numbers=interpolated_atoms_list[interpolation_i],
                    pbc=True,
                )
                if interpolation_i == 0:
                    write(f'{args.output_folder}rocksalt_{metal}.cif', interpolated_structure)
                elif interpolation_i == len(interp_step_list) - 1:
                    write(f'{args.output_folder}spinel_{metal}.cif', interpolated_structure)
                else:
                    write(f'{args.output_folder}interpolated_rocksalt_to_spinel_step{interp_step_list[interpolation_i]}_sample{sample_i_list[interpolation_i]}_{metal}.cif', interpolated_structure)
                
            # Generate zincblende to spinel interpolations
            interp_step_list, sample_i_list, interpolated_cell_parameters_list, interpolated_atoms_list, interpolated_positions_list = zincblende_to_spinel(zincblende_transformed, spinel_transformed, interpolation_steps=args.interpolation_steps, atom_samples=args.atom_samples, identification_threshold=args.identification_threshold)

            # Check for and remove duplicate structures
            duplicate_indices = []
            for i in range(len(interp_step_list)):
                for j in range(i, len(interp_step_list)):
                    if i == j:
                        continue
                    if np.array_equal(interpolated_positions_list[i], interpolated_positions_list[j]) and np.array_equal(interpolated_cell_parameters_list[i], interpolated_cell_parameters_list[j]):
                        if j == len(interp_step_list) - 1:
                            duplicate_indices.append(i)
                        else:
                            duplicate_indices.append(j)

            interp_step_list = [interp_step_list[i] for i in range(len(interp_step_list)) if i not in duplicate_indices]
            sample_i_list = [sample_i_list[i] for i in range(len(sample_i_list)) if i not in duplicate_indices]
            interpolated_cell_parameters_list = [interpolated_cell_parameters_list[i] for i in range(len(interpolated_cell_parameters_list)) if i not in duplicate_indices]
            interpolated_atoms_list = [interpolated_atoms_list[i] for i in range(len(interpolated_atoms_list)) if i not in duplicate_indices]
            interpolated_positions_list = [interpolated_positions_list[i] for i in range(len(interpolated_positions_list)) if i not in duplicate_indices]

            # Save interpolations as cif files
            for interpolation_i in range(len(interp_step_list)):
                interpolated_structure = Atoms(
                    cell=interpolated_cell_parameters_list[interpolation_i],
                    scaled_positions=interpolated_positions_list[interpolation_i],
                    numbers=interpolated_atoms_list[interpolation_i],
                    pbc=True,
                )
                if interpolation_i == 0:
                    write(f'{args.output_folder}zincblende_{metal}.cif', interpolated_structure)
                elif interpolation_i == len(interp_step_list) - 1:
                    write(f'{args.output_folder}spinel_{metal}.cif', interpolated_structure)
                else:
                    write(f'{args.output_folder}interpolated_zincblende_to_spinel_step{interp_step_list[interpolation_i]}_sample{sample_i_list[interpolation_i]}_{metal}.cif', interpolated_structure)
    elif args.interpolation_type == 'nickelArsenide_to_cadmiumIodide':
        # Get the relevant CIF files
        
        # Nickel Arsenide structure
        nickelArsenide_structures = [str(filepath) for filepath in Path('./data/CIFs/').rglob('NickelArsenide*.cif')]
        nickelArsenide_compositions = [filepath.split('/')[-1].split('_')[-1].split('.')[0] for filepath in nickelArsenide_structures]
        nickelArsenide_metals = [composition[:-1] for composition in nickelArsenide_compositions]

        # Cadmium Iodide structure
        cadmiumIodide_structures = [str(filepath) for filepath in Path('./data/CIFs/').rglob('CadmiumIodide*.cif')]
        cadmiumIodide_compositions = [filepath.split('/')[-1].split('_')[-1].split('.')[0] for filepath in cadmiumIodide_structures]
        cadmiumIodide_metals = [composition[:-2] for composition in cadmiumIodide_compositions]
    
        # Generate the interpolated structures
        for metal in tqdm(nickelArsenide_metals, desc='Generating interpolated structures'):
            # Read the structures
            # Nickel Arsenide
            nickelArsenide_structure = read(nickelArsenide_structures[nickelArsenide_metals.index(metal)])
            nickelArsenide_transformed = nickelArsenide_transformation(nickelArsenide_structure, expansion_factor=2, translation=0.0, inversion=False)
            
            # Cadmium Iodide
            cadmiumIodide_structure = read(cadmiumIodide_structures[cadmiumIodide_metals.index(metal)])
            cadmiumIodide_transformed = cadmiumIodide_transformation(cadmiumIodide_structure, expansion_factor=2, translation=0.0, inversion=False)
            
            # Generate nickelArsenide to cadmiumIodide interpolations
            interp_step_list, sample_i_list, interpolated_cell_parameters_list, interpolated_atoms_list, interpolated_positions_list = nickelArsenide_to_cadmiumIodide(nickelArsenide_transformed, cadmiumIodide_transformed, identification_threshold=0.1)

            # Check for and remove duplicate structures
            duplicate_indices = []
            for i in range(len(interp_step_list)):
                for j in range(i, len(interp_step_list)):
                    if i == j:
                        continue
                    if np.array_equal(interpolated_positions_list[i], interpolated_positions_list[j]) and np.array_equal(interpolated_cell_parameters_list[i], interpolated_cell_parameters_list[j]):
                        if j == len(interp_step_list) - 1:
                            duplicate_indices.append(i)
                        else:
                            duplicate_indices.append(j)

            interp_step_list = [interp_step_list[i] for i in range(len(interp_step_list)) if i not in duplicate_indices]
            sample_i_list = [sample_i_list[i] for i in range(len(sample_i_list)) if i not in duplicate_indices]
            interpolated_cell_parameters_list = [interpolated_cell_parameters_list[i] for i in range(len(interpolated_cell_parameters_list)) if i not in duplicate_indices]
            interpolated_atoms_list = [interpolated_atoms_list[i] for i in range(len(interpolated_atoms_list)) if i not in duplicate_indices]
            interpolated_positions_list = [interpolated_positions_list[i] for i in range(len(interpolated_positions_list)) if i not in duplicate_indices]

            print(f"Removed {len(duplicate_indices)} duplicate structures")


            # Save interpolations as cif files
            for interpolation_i in range(len(interp_step_list)):
                interpolated_structure = Atoms(
                    cell=interpolated_cell_parameters_list[interpolation_i],
                    scaled_positions=interpolated_positions_list[interpolation_i],
                    numbers=interpolated_atoms_list[interpolation_i],
                    pbc=True,
                )
                if interpolation_i == 0:
                    write(f'{args.output_folder}NickelArsenide_{metal}.cif', interpolated_structure)
                elif interpolation_i == len(interp_step_list) - 1:
                    write(f'{args.output_folder}CadmiumIodide_{metal}.cif', interpolated_structure)
                else:
                    write(f'{args.output_folder}interpolated_NickelArsenide_to_CadmiumIodide_step{interp_step_list[interpolation_i]}_sample{sample_i_list[interpolation_i]}_{metal}.cif', interpolated_structure)