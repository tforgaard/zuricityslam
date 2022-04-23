#!/bin/bash
BASE=/cluster/project/infk/courses/252-0579-00L/group07
PAIRING=sequential+retrieval
DATASET=4k/long_walk_zurich

mkdir -p ./logs/${DATASET}
source ./scripts/colmap_startup.sh

mkdir -p ${BASE}/outputs/${DATASET}_${PAIRING}_fps2/sfm_superpoint+superglue/colmap_hier2/snap
bsub -o ./logs/${DATASET}/test_${PAIRING}_fps2_colmap_mapping_hier.out -W 24:00 -n 32 -R "rusage[mem=4096,ngpus_excl_p=1]" "colmap hierarchical_mapper --database_path ${BASE}/outputs/${DATASET}_${PAIRING}_fps2/sfm_superpoint+superglue/database.db --image_path ${BASE}/datasets/${DATASET}/images-fps2 --output_path ${BASE}/outputs/${DATASET}_${PAIRING}_fps2/sfm_superpoint+superglue/colmap_hier  --leaf_max_num_images 1000 --num_workers 5 --image_overlap 50 --Mapper.num_threads 32 --Mapper.init_min_tri_angle 8 --Mapper.ba_global_use_pba=1 --Mapper.ba_global_pba_gpu_index=0 --Mapper.snapshot_path ${BASE}/outputs/${DATASET}_${PAIRING}_fps2/sfm_superpoint+superglue/colmap_hier2/snap --Mapper.snapshot_images_freq 500 --Mapper.tri_ignore_two_view_tracks 0"