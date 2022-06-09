#!/bin/bash
#BSUB -J hloc_pipeline
#BSUB -n 32
#BSUB -R rusage[mem=4096,ngpus_excl_p=2]
#BSUB -W 24:00             
####### -o .logs/hloc_pipeline_%J.log

BASE=/cluster/project/infk/courses/252-0579-00L/group07
PAIRING=sequential+retrieval
VIDEO=W25QdyiFnh0

IMAGES=${BASE}/datasets/images/${VIDEO}
OUTPUT=${BASE}/outputs/${VIDEO}_${PAIRING}

# Load required modules and variables for using colmap
source ./scripts/colmap_startup.sh

# Run hloc pipeline
python3 -m cityslam.mapping.reconstruction      --dataset ${IMAGES} \
                                                --outputs ${OUTPUT} \
                                                --num_loc 7 \
                                                --window_size 6 \
                                                --pairing ${PAIRING}
                                                --run_reconstruction 1