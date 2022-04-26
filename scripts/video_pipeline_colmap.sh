#!/bin/bash

BASE=/cluster/project/infk/courses/252-0579-00L/group07
PAIRING=sequential+retrieval
VIDEO=W25QdyiFnh0

# Create log directory if it does not exist
mkdir -p ./logs/${VIDEO}

# Run hloc pipeline, extracting features, pairing, matching...
bsub -J hloc_pipeline -o ./logs/${VIDEO}/${PAIRING}_%J.out < ./scripts/single_video_pipeline.sh

# Run faster hierarchical mapper for reconstruction
bsub -J hierarchical_mapper -w "ended(hloc_pipeline)" -o ./logs/${VIDEO}/${PAIRING}_colmap_hier_mapping_%J.out < ./scripts/colmap_hier_mapper.sh