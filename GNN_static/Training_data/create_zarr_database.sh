#!/usr/bin/bash
#SBATCH --job-name=train36
#SBATCH --qos=nf
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB
##SBATCH --output=carra2.%j.out
##SBATCH --error=carra2.%j.out

cd /perm/sbjb/Projects/CERISE/git/CERISE_obs_op_static-GNN/GNN_static/Training_data

module load python3/3.10.10-01

python3 create_zarr_database.py
#python3 nn_test.py