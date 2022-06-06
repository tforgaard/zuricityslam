#!/bin/bash
#BSUB -J hierarchical_mapper
#BSUB -n 32
#BSUB -R rusage[mem=4096,ngpus_excl_p=1]
#BSUB -W 24:00             
####### -o .logs/hierarchical_mapper_%J.log

BASE=/cluster/project/infk/courses/252-0579-00L/group07
PAIRING=sequential+retrieval
VIDEO=W25QdyiFnh0

MODEL_DIR=${BASE}/outputs/${VIDEO}_${PAIRING}
IMAGES=${BASE}/datasets/images/${VIDEO}
OUTPUT=${MODEL_DIR}/colmap_hier

# Load required modules and variables for using colmap
source ./scripts/colmap_startup.sh

# Create output directories if they do not exist
mkdir -p ${OUTPUT}/snap

# Run hierarchical mapper
colmap hierarchical_mapper  --database_path ${MODEL_DIR}/database.db \
                            --image_path ${IMAGES} \
                            --output_path ${OUTPUT} \
                            --num_workers 4 \
                            --image_overlap 100 \
                            --leaf_max_num_images 1000 \
                            --Mapper.num_threads 32 \
                            --Mapper.ba_global_use_pba 1 \
                            --Mapper.init_min_tri_angle 8 \
                            --Mapper.snapshot_images_freq 500 \
                            --Mapper.ba_global_pba_gpu_index -1 \
                            --Mapper.snapshot_path ${OUTPUT}/snap \
                            --Mapper.tri_ignore_two_view_tracks 0 \