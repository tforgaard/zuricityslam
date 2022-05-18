from pathlib import Path
import os
from datetime import datetime

import cityslam
from cityslam.videointerface import videointerface, downloader
from cityslam.preprocessing import preprocessing
#from cityslam.mapping import single_video_pipeline
#from cityslam.utils import visualization

os.umask(0o002)

base_dir = Path('/cluster/project/infk/courses/252-0579-00L/group07')

videos_path = base_dir / 'datasets' / 'videos'
images_path = base_dir / 'datasets' / 'images'
queries_base_path = base_dir / 'datasets' / 'queries'
# apireturns_path = base_dir / 'datasets' / 'apireturns'
output_path = base_dir / 'outputs' / 'models'

num_vids = 1

# Demo of complete pipeline
# Things to be aware of:
# format="wv" means find videos with worst quality, useful for debugging
# duration='00:03:00' means that we only convert the first three minutes of the video to images, useful for debugging
# set overwrite to True if you want to ignore cached files

# Fetch videos for query
# apireturns_path = Path(apireturns_path.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
queries_path = Path(queries_base_path.stem + "stored.json")
blacklist_path = Path(queries_base_path.stem + "blacklist.json")
video_ids = videointerface.main(queries_path, blacklist_path, "coordinates", "47.371667, 8.542222", max_results=25, overwrite=False, verbose=False)

# Download videos
downloader.main(videos_path, queries_path, format="wv")
exit(0)

# Split videos into frames
image_folders = preprocessing.main(
    videos_path, images_path, num_vids=num_vids, overwrite=False, fps=2,  duration='00:03:00')

# run sfm on videos
for image_folder in image_folders:
    sfm_path = output_path / image_folder.name
    reconstruction = single_video_pipeline.main(
        image_folder, sfm_path, window_size=4, num_loc=4, pairing='sequential+retrieval', run_reconstruction=True)

    # Visualize feature points
    if reconstruction is not None:
        print("creating sfm video")
        visualization.visualize_sfm_2d_video(
            reconstruction, image_folder, output=sfm_path, color_by='depth', dpi=150, del_tmp_frames=True)
