#!/usr/bin/bash
#SBATCH --job-name=conv18val
#SBATCH --qos=nf
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=100GB

module load python3/3.10.10-01

python3 /perm/sbjb/Projects/CERISE/git/CERISE_obs_op_static-GNN/GNN_static/Training_data/Convert_hdf_to_zarr_static.py
