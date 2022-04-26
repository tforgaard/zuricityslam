#!/bin/bash
#BSUB -J mapper
#BSUB -n 32
#BSUB -R rusage[mem=4096,ngpus_excl_p=1]
#BSUB -W 24:00             
####### -o .logs/mapper_%J.log

BASE=/cluster/project/infk/courses/252-0579-00L/group07
PAIRING=sequential+retrieval
VIDEO=W25QdyiFnh0

MODEL_DIR=${BASE}/outputs/${VIDEO}_${PAIRING}/sfm_sp+sg
IMAGES=${BASE}/datasets/images/${VIDEO}
OUTPUT=${MODEL_DIR}/colmap

# Load required modules and variables for using colmap
source ./scripts/colmap_startup.sh

# Create output directories if they do not exist
mkdir -p ${OUTPUT}/snap

# Run mapper
colmap mapper   --database_path ${MODEL_DIR}/database.db \
                --image_path ${IMAGES} \
                --output_path ${OUTPUT} \
                --Mapper.num_threads 32 \
                --Mapper.ba_global_use_pba 1 \
                --Mapper.init_min_tri_angle 8 \
                --Mapper.ba_global_pba_gpu_index -1 \
                --Mapper.snapshot_images_freq 500 \
                --Mapper.snapshot_path ${OUTPUT}/snap \
                --Mapper.tri_ignore_two_view_tracks 0 \