#!/bin/bash
#SBATCH --job-name=test_4d_split_ls_super_cell

#SBATCH --ntasks=1 --cpus-per-task=12 --mem=8000M

#SBATCH -p gpu --gres=gpu:titanrtx:1

#SBATCH --time=0-2:00:00

#SBATCH -o ./slurm_outputs/scvae-%j.out #STDOUT

hostname
echo $CUDA_VISIBLE_DEVICES

python train.py --setup_json test_setup.json

python test.py --test_data validation --setup_json ./models/test_4d_split_ls_super_cell/setup_json.json
