#!/bin/bash
BASE=/cluster/project/infk/courses/252-0579-00L/group07
PAIRING=sequential+retrieval
DATASET=4k/loop_walk_zurich

mkdir -p ./logs/${DATASET}
source ./scripts/colmap_startup.sh

bsub -J hloc_pipeline -o ./logs/${DATASET}/test_${PAIRING}_fps2.out -W 2:00 -n 16 -R "rusage[mem=4096,ngpus_excl_p=2]" "python3 src/mapping/single_video_pipeline.py --dataset datasets/${DATASET} --outputs outputs/${DATASET}_${PAIRING}_fps2 --num_loc 5 --window_size 5 --pairing ${PAIRING}"
bsub -J colmap_mapping -w "ended(hloc_pipeline)" -o ./logs/${DATASET}/test_${PAIRING}_fps2_colmap_mapping.out -W 12:00 -n 16 -R "rusage[mem=4096,ngpus_excl_p=1]" "mkdir -p ${BASE}/outputs/${DATASET}_${PAIRING}_fps2/sfm_superpoint+superglue/colmap && colmap mapper --database_path ${BASE}/outputs/${DATASET}_${PAIRING}_fps2/sfm_superpoint+superglue/database.db --image_path ${BASE}/datasets/${DATASET}/images-fps2 --output_path ${BASE}/outputs/${DATASET}_${PAIRING}_fps2/sfm_superpoint+superglue/colmap --Mapper.num_threads 16 --Mapper.ba_global_use_pba=1 --Mapper.ba_global_pba_gpu_index=0"