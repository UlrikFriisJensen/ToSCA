#!/bin/bash
#SBATCH --job-name=Test_run_norm_low_lr

#SBATCH --ntasks=1 --cpus-per-task=8 --mem=6000M

#SBATCH -p gpu --gres=gpu:titanrtx:1

#SBATCH --time=1-00:00:00

#SBATCH -o ./slurm_outputs/scvae-%j.out #STDOUT

hostname
echo $CUDA_VISIBLE_DEVICES

python train.py --setup_json test_setup.json

python test.py --test_data validation --setup_json ./models/Test_run_norm_low_lr/setup_json.json