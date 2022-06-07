#!/bin/bash
#BSUB -J find_transitions
#BSUB -n 8
#BSUB -W 4:00  
#BSUB -R rusage[mem=4096,ngpus_excl_p=1]
#BSUB -R select[gpu_mtotal0>=4096]
#####BSUB -o cluster/project/infk/courses/252-0579-00L/group07/kriss/logs/dl_vid%J.out


source ./scripts/colmap_startup.sh

BASE=/cluster/project/infk/courses/252-0579-00L/group07
VIDS_PATH=${BASE}/datasets/videos_wv
TRANS_PATH=${BASE}/datasets/transitions
CROPP_PATH=${BASE}/datasets/transitions_cropped

VIDEO_IDS="W25QdyiFnh0 s5MlvdGL8Lg fGxbtg1ytJo uVCG0MAkW0E ctYA9Fzo6WM vl_BfAZf4LY KzrCHdviKho yTSf9vI8nI4 DdxEr70jC6E Ka3Ss3cdk0Y N1FXwkV0MMY _NmYvuEILw4 m3EbUAoyWHo lN4j2iiFpgQ ITntTt4qkWY dzJV1iG8C3A 2obsKLoZQdU EctFmZiZiEs z6q7PG_pTEs jsOKs7cnoI4 73IVzh0R-Lo NpQqe_nLWjg Rq2VnLu5aTw 207zUTmTiBc t1DpmdNq6QA Jm3eNoYT_vY _k_mojbxh_A ZQRpwAJPfQo xaMI2KnNS4Y 28vkHTRVeqo tENRFDzrim4 jjk1IEryf7Q zjGlPabjZPE ITo-ZFQWSYM CFanF4f0fs4 OSMTFYGVLyE KjEo9sf5y-I ENDo7IPlgMg w4aMLRGpMEw NFGvl2XeOxI QGCTwz1Kqe0 _jJGc4r1mzk ziTGhoM0SOU 453esepQAmQ P6Rc9zao3wo cdzf6Tw7bfA 9Xr-rNyLIjk 606xkPY56R4 ElVdGIWdteI ZpJmSnOTKG0 tmDFCotvcoQ K2rlPMRUYKI"

#we run the transition finder on low quality versions for speedup
#python3 -m cityslam.videointerface.downloader   ${VIDEO_IDS} \
#                                                --output ${VIDS_PATH} \
#                                                --format 'wv'

python3 -m cityslam.preprocessing.transitions   --videos_dir ${VIDS_PATH} \
                                                --video_ids  ${VIDEO_IDS} \
                                                --output_transitions ${TRANS_PATH} \
                                                --output_cropped ${CROPP_PATH} \

                                                   
                                       
