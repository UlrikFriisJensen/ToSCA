#!/bin/bash
#SBATCH --job-name=More_gnn_heads

#SBATCH --ntasks=1 --cpus-per-task=8 --mem=6000M

#SBATCH -p gpu --gres=gpu:titanrtx:1

#SBATCH --time=1-00:00:00

#SBATCH -o ./slurm_outputs/scvae-%j.out #STDOUT

hostname
echo $CUDA_VISIBLE_DEVICES

python train.py --setup_json test_setup.json

python test.py --test_data validation --setup_json ./models/More_gnn_heads/setup_json.json