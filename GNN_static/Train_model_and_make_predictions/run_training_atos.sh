#!/usr/bin/bash
#SBATCH --job-name=GNN_18GHz
#SBATCH --qos=ng
#SBATCH --gpus=1
#SBATCH --time=06:00:00

##SBATCH --cpus-per-task=1
#SBATCH --mem=24GB
#SBATCH --output=/home/sbjb/tmp/ml_train36.%j.out
#SBATCH --error=/home/sbjb/tmp/ml_train36.%j.out

cd /perm/sbjb/Projects/CERISE/git/cerise_PR
module load cuda
source /etc/ecmwf/nfs/dh2_perm_b/sbjb/Projects/CERISE/git/cerise_PR/.venv/bin/activate

cd /perm/sbjb/Projects/CERISE/git/CERISE_obs_op_static-GNN/GNN_static/Train_model_and_make_predictions

#python3 Long_training_GNN.py 20
python3 Train_GNN.py
