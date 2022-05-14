
#source ./scripts/colmap_startup.sh

#module load gcc/8.2.0 cuda/11.2.2 cudnn/8.2.4.15 ffmpeg
#conda install tensorflow
#bsub -I -n 1 -W 1:00 -R "rusage[mem=2048, ngpus_excl_p=1]" 'python -c ""'

bsub -n 16 -W 1:00 -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=4096]" python cityslam/videointerface/transition_cut.py

