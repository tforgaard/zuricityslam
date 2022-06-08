#!/bin/bash
#BSUB -J reconstruction
#BSUB -n 32
#BSUB -R rusage[mem=1024]
#BSUB -W 24:00             

# Load required modules and variables for using colmap
source ./scripts/colmap_startup.sh

# Run hloc pipeline
python3 pipelines/pipeline_reconstruction.py