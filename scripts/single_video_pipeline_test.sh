#!/bin/bash
bsub -o short_walk_zurich_test.out -n 8 -R "rusage[mem=4096,ngpus_excl_p=1]" "python3 src/mapping/single_video_pipeline.py 
                                    \ --dataset datasets/short_walk_zurich --outputs outputs/short_walk_zurich --num_loc 5"