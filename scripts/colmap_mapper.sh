#!/bin/bash
BASE=/cluster/project/infk/courses/252-0579-00L/group07
PAIRING=sequential+retrieval
DATASET=4k/loop_walk_zurich

mkdir -p ./logs/${DATASET}
source ./scripts/colmap_startup.sh
#TODO add snapshots!
mkdir -p ${BASE}/outputs/${DATASET}_${PAIRING}_fps2/sfm_superpoint+superglue/colmap
bsub -o ./logs/${DATASET}/test_${PAIRING}_fps2_colmap_mapping.out -W 12:00 -n 16 -R "rusage[mem=4096,ngpus_excl_p=1]" "colmap mapper --database_path ${BASE}/outputs/${DATASET}_${PAIRING}_fps2/sfm_superpoint+superglue/database.db --image_path ${BASE}/datasets/${DATASET}/images-fps2 --output_path ${BASE}/outputs/${DATASET}_${PAIRING}_fps2/sfm_superpoint+superglue/colmap --Mapper.num_threads 16 --Mapper.ba_global_use_pba=1 --Mapper.ba_global_pba_gpu_index=0"