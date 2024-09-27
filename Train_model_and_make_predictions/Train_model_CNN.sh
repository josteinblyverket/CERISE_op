#!/bin/bash -f
#$ -N Train_patch_model_CERISE
#$ -l h_rt=24:00:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=12G,mem_free=12G,h_data=12G
#$ -q gpu-r8.q
#$ -l h=gpu-03.ppi.met.no
##$ -j y
#$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y

module use /modules/MET/rhel8/user-modules/
module load cuda/11.6.0

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate /lustre/storeB/users/cyrilp/mycondaTF

python3 "/lustre/storeB/users/cyrilp/CERISE/Scripts/Patch_CNN/Models/CNN/Train_model_CNN.py"
#rm /home/cyrilp/Documents/ERR/ERR_Run_UNet_COSI.* /home/cyrilp/Documents/OUT/OUT_Run_UNet_COSI.*
