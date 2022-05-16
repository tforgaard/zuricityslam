#!/bin/bash


source ./scripts/colmap_startup.sh

BASE=/cluster/project/infk/courses/252-0579-00L/group07
QUERY_PATH=${BASE}/datasets/queries

QUERY_TYPE="coordinates"
QUERY="47.371667, 8.542222"
N_VIDS="25"

python3 -m cityslam.videointerface.videointerface   --queries_path ${QUERY_PATH} \
                                                    --input_type ${QUERY_TYPE} \
                                                    --query "${QUERY}" \
                                                    --num_vids ${N_VIDS} # \
                                                    # --overwrite
