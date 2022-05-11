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
output_path = base_dir / 'outputs' / 'models'

num_vids = 5
overwrite = False

# Demo of complete pipeline
# Things to be aware of:
# format="wv" means find videos with worst quality, useful for debugging
# duration='00:03:00' means that we only convert the first three minutes of the video to images, useful for debugging
# set overwrite to True if you want to ignore cached files

# Fetch videos for query
video_ids = videointerface.main(queries_path, "coordinates", "47.371667, 8.542222",
                                num_vids=num_vids, overwrite=False, verbose=True)

# if you know what videos you want to download just overwrite video ids, i.e.
# video_ids = ['gTHMvU3XHBk'], 'TZIHy1cZJ-U']
# these to specific video ids, are merge testing videos

# Download videos
downloader.main(videos_path, video_ids, format="bv", overwrite=overwrite)

# Split videos into frames
image_folders = preprocessing.main(
    videos_path, images_path, video_ids, overwrite=overwrite, fps=2)

# run sfm on videos
for image_folder in image_folders:
    sfm_path = output_path / image_folder.name
    reconstruction = single_video_pipeline.main(
        image_folder, sfm_path, window_size=6, num_loc=6, pairing='sequential+retrieval', run_reconstruction=True)

    # Visualize feature points
    if reconstruction is not None:
        print("creating sfm video")
        visualization.visualize_sfm_2d_video(
            reconstruction, image_folder, output=sfm_path, color_by='depth', dpi=150, del_tmp_frames=True)
