#!/bin/bash
#SBATCH --job-name=Interpolation_Supercell_latentLog_2d_latentMSE_smallerEncoder

#SBATCH --ntasks=1 --cpus-per-task=12 --mem=8000M

#SBATCH -p gpu --gres=gpu:1

#SBATCH --time=2-00:00:00

#SBATCH -o ./slurm_outputs/interp-scvae-%j.out #STDOUT

hostname
echo $CUDA_VISIBLE_DEVICES

python train.py --setup_json test_setup.json

python test.py --test_data validation --setup_json ./models/Interpolation_Supercell_latentLog_2d_latentMSE_smallerEncoder/setup_json.json

# python test_experimentalData.py --data_folder ./data/Experimental/Jens/ --setup_json ./models/Combined_data_run_3d/setup_json.json
# python test_interpolation.py --setup_json ./models/Combined_data_run_3d/setup_json.json