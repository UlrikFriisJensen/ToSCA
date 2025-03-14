#!/bin/bash
#SBATCH --job-name=Simulate_interpolations

#SBATCH --array=1-10%4

#SBATCH --ntasks=1 --cpus-per-task=12 --mem=12000M

#SBATCH -p gpu --gres=gpu:titanrtx:1

#SBATCH --time=1-00:00:00

#SBATCH -o ./slurm_outputs/gen-%A-%a.out #STDOUT

hostname
echo $CUDA_VISIBLE_DEVICES

python ./modules/h5_constructor.py --batch ./batch_$SLURM_ARRAY_TASK_ID.txt --output ./data/h5/Interpolations/

