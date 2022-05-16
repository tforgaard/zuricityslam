#!/bin/bash
#BSUB -J generate_frames
#BSUB -n 16
#BSUB -R rusage[mem=4096]
#BSUB -W 24:00     
#BSUB -outdir ./logs
#BSUB -o ./logs/gen_frames%J.out

source ./scripts/colmap_startup.sh

BASE=/cluster/project/infk/courses/252-0579-00L/group07
VIDS_PATH=${BASE}/datasets/videos
IMGS_PATH=${BASE}/datasets/images

FPS=2

# If you know what video ids you want to download just set 
# VIDEO_IDS="da3rflkn 23dklknls alj12dmk"

python3 -m cityslam.preprocessing.preprocessing --videos ${VIDS_PATH} \
                                                --output ${IMGS_PATH} \
                                                --fps ${FPS}
                                                # --video_ids ${VIDEO_IDS} \
                                                # --overwrite
