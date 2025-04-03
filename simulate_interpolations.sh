#!/bin/bash
#SBATCH --job-name=Simulate_interpolations

#SBATCH --ntasks=1 --cpus-per-task=12 --mem=8000M

#SBATCH --time=0-02:00:00

#SBATCH -o ./slurm_outputs/sim-%j.out #STDOUT

hostname

# python ./modules/generate_cifs.py --dataset ./data/

python ./modules/generate_interpolation_cifs.py --cif_folder ./data/CIFs/CHILI-3K/ --output_folder ./data/CIFs/Interpolations_v2/ --interpolation_type nickelArsenide_to_cadmiumIodide --interpolation_steps 5 --atom_samples 3

python ./modules/cif_batching.py --dataset ./data/CIFs/Interpolations_v2/ --batch_size 10