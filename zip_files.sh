#!/bin/bash
#SBATCH --job-name=Zip_dataset

#SBATCH --ntasks=1 --cpus-per-task=12 --mem=8G

#SBATCH --time=0-03:00:00

#SBATCH -o ./slurm_outputs/zip-%j.out #STDOUT

hostname

python ./zip_data.py --data_folder ./data/h5/Interpolations_small/ --output_file ./data/CHILI-Interpolation_small 

