#!/bin/bash
BASE=/cluster/project/infk/courses/252-0579-00L/group07
PAIRING=sequential+retrieval
DATASET=4k/long_walk_zurich

# Create log directory if it does not exist
mkdir -p ./logs/${DATASET}

# Run hloc pipeline, extracting features, pairing, matching...
bsub -J hloc_pipeline -o ./logs/${DATASET}/test_${PAIRING}_fps2_%J.out < ./scripts/single_video_pipeline.sh

# Run faster hierarchical mapper for reconstruction
bsub -J hierarchical_mapper -w "ended(hloc_pipeline)" -o ./logs/${DATASET}/test_${PAIRING}_fps2_colmap_hier_mapping_%J.out < ./scripts/colmap_hier_mapper.sh