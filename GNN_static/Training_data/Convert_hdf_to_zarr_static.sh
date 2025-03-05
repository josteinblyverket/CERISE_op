#!/bin/bash -f
#$ -N Create_zarr_dataset
#$ -l h_rt=24:00:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=50G,mem_free=50G,h_data=50G
#$ -q bigmem-r8.q
#$ -t 1-1
##$ -j y
#$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate production-08-2024

python3 /lustre/storeB/users/cyrilp/CERISE/Scripts/GNN/Training_data_GNN/Static_graphs/Convert_hdf_to_zarr_static.py
