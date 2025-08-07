#!/bin/bash -f

#SBATCH --job-name=NNPred18
#SBATCH --qos=nf
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB

export PATH="/etc/ecmwf/nfs/dh2_perm_b/sbjb/Projects/CERISE/git/cerise_PR/.venv/bin:$PATH"
export PYTHONPATH="/perm/sbjb/Projects/CERISE/git/cerise_PR/.venv/lib/python3.10/site-packages:$PYTHONPATH"
cd /home/sbjb/perm/Projects/CERISE/git/CERISE_obs_op_static-GNN/NN

python3 Predict_NN.py
