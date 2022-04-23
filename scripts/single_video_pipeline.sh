#!/bin/bash
#BSUB -J hloc_pipeline
#BSUB -n 32
#BSUB -R rusage[mem=4096,ngpus_excl_p=2]
#BSUB -W 24:00             
####### -o .logs/hloc_pipeline_%J.log

PAIRING=sequential+retrieval
DATASET=4k/long_walk_zurich

# Load required modules and variables for using colmap
source ./scripts/colmap_startup.sh

# Run hloc pipeline
python3 src/mapping/single_video_pipeline.py    --dataset datasets/${DATASET} \
                                                --outputs outputs/${DATASET}_${PAIRING}_fps2 \
                                                --num_loc 7 \
                                                --window_size 6 \
                                                --pairing ${PAIRING}
                                                --run_reconstruction 1