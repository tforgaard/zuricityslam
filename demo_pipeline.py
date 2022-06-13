from pathlib import Path
import os

from cityslam.videointerface import videointerface, downloader
from cityslam.preprocessing import preprocessing, transitions, create_img_list
from cityslam.mapping import single_video_pipeline
from cityslam.utils import visualization
from natsort import natsorted

# setup necessary paths
base_dir = Path('./demo')
base_dir.parent.mkdir(exist_ok=True, parents=True)

videos_path = base_dir / 'datasets' / 'videos'
images_path = base_dir / 'datasets' / 'images'
queries_path = base_dir / 'datasets' / 'queries'
output_path = base_dir / 'outputs' / 'models'
output_transitions = base_dir / 'outputs' / 'transitions'
output_transitions_cropped = base_dir / 'outputs' / 'transitions_cropped'
image_list_path = base_dir / 'outputs' / 'image_list'

overwrite = False
video_ids = ['gTHMvU3XHBk', 'TZIHy1cZJ-U']
fps = 2


# Download videos
downloader.main(videos_path, video_ids, format="bv", overwrite=overwrite)

# Split into frames
image_folders = preprocessing.main(
    videos_path, images_path, video_ids, overwrite=overwrite, fps=fps)

# Find transitions and split into frames
path_to_weights = Path('./cityslam/preprocessing')

transitions.main(
    videos_path, video_ids, path_to_weights, output_transitions, output_transitions_cropped, 5*60, 10, fps, 0.5, overwrite_cuts=True)