#!/bin/bash
#BSUB -J download_vids
#BSUB -n 4
#BSUB -W 24:00             
####### -o .logs/dl_vid%J.log

module load eth_proxy

mkdir -p ./logs/dl_vid

BASE=/cluster/project/infk/courses/252-0579-00L/group07/kriss
VIDS_PATH=${BASE}/datasets/videos
IMGS_PATH=${BASE}/datasets/images
QUERY_PATH=${BASE}/datasets/queries

QUERY_TYPE="coordinates"
QUERY="47.371667, 8.542222"
QUALITY="bv"
N_VIDS="25"

python3 cityslam/videointerface/videointerface.py   --videos_path ${VIDS_PATH} \
                                                    --images_path ${IMGS_PATH} \
                                                    --queries_path ${QUERY_PATH} \
                                                    --input_type ${QUERY_TYPE} \
                                                    --query "${QUERY}" \
                                                    --format ${QUALITY} \
                                                    --num_vids ${N_VIDS}
                                       
