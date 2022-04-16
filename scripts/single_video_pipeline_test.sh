#!/bin/bash
PAIRING=sequential+retrieval
bsub -o short_walk_zurich_test_${PAIRING}_fps2.out -n 16 -R "rusage[mem=4096,ngpus_excl_p=2]" "python3 src/mapping/single_video_pipeline.py --dataset datasets/short_walk_zurich --outputs outputs/short_walk_zurich_${PAIRING}_fps2 --num_loc 5 --window_size 5 --pairing ${PAIRING}"
