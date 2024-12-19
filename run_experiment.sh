#!/bin/bash
#SBATCH --job-name=4d_split_ls_super_cell_unnormalizedCellParameters

#SBATCH --ntasks=1 --cpus-per-task=12 --mem=8000M

#SBATCH -p gpu --gres=gpu:titanrtx:1

#SBATCH --time=0-13:00:00

#SBATCH -o ./slurm_outputs/scvae-%j.out #STDOUT

hostname
echo $CUDA_VISIBLE_DEVICES

python train.py --setup_json test_setup.json

python test.py --test_data validation --setup_json ./models/4d_split_ls_super_cell_unnormalizedCellParameters/setup_json.json
