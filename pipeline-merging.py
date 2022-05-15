from pathlib import Path
import os

from cityslam.videointerface import videointerface, downloader
from cityslam.preprocessing import preprocessing
from cityslam.mapping import single_video_pipeline
from cityslam.utils import visualization

# fix file permission problems
os.umask(0o002)

# setup necessary paths
base_dir = Path('/cluster/project/infk/courses/252-0579-00L/group07')

videos_path = base_dir / 'datasets' / 'videos'
images_path = base_dir / 'datasets' / 'images'
queries_path = base_dir / 'datasets' / 'queries'
output_path = base_dir / 'outputs' / 'models-merge-testing3'

overwrite = False
video_ids = ['gTHMvU3XHBk', 'TZIHy1cZJ-U']

# Download videos
downloader.main(videos_path, video_ids, format="bv", overwrite=overwrite)

# Split videos into frames
image_folders = preprocessing.main(
    videos_path, images_path, video_ids, overwrite=overwrite, fps=2)

# run sfm on videos
for video_id in video_ids:

    # sfm_path = output_path / image_folder.name
    reconstruction = single_video_pipeline.main(
        images_path, output_path, video_id=video_id, window_size=6, num_loc=6, pairing='sequential+retrieval', run_reconstruction=False)

    # Visualize feature points
    if reconstruction is not None:
        sfm_path = output_path / video_id
        print("creating sfm video")
        visualization.visualize_sfm_2d_video(
            reconstruction, images_path, output=sfm_path, color_by='depth', dpi=150, del_tmp_frames=True)
