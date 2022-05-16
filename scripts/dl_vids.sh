#!/bin/bash
#BSUB -J download_vids
#BSUB -n 4
#BSUB -R rusage[mem=4096]
#BSUB -W 4:00     
#BSUB -outdir ./logs
#BSUB -o ./logs/dl_vid%J.out

source ./scripts/colmap_startup.sh

BASE=/cluster/project/infk/courses/252-0579-00L/group07
VIDS_PATH=${BASE}/datasets/videos

QUALITY="bv"

VIDEO_IDS=$(./scripts/find_vids.sh)
echo "Found videos:"
echo ${VIDEO_IDS}
# If you know what video ids you want to download
# just set VIDEO_IDS="da3rflkn 23dklknls alj12dmk"

python3 -m cityslam.videointerface.downloader   ${VIDEO_IDS} \
                                                --output ${VIDS_PATH} \
                                                --format ${QUALITY} \
                                                # --overwrite
