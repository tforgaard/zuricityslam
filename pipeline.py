from pathlib import Path

from cityslam.videointerface import videointerface
from cityslam.mapping import single_video_pipeline
from cityslam.utils import visualization


base_dir = Path('/cluster/project/infk/courses/252-0579-00L/group07')

videos_path = base_dir / 'datasets' / 'videos'
images_path = base_dir / 'datasets' / 'images'
querie_path = base_dir / 'datasets' / 'queries'
output_path = base_dir / 'outputs'


# Demo of complete pipeline
# Things to be aware of:
# format="wv" means find videos with worst quality, useful for debugging
# duration='00:03:00' means that we only convert the first three minutes of the video to images, useful for debugging
# set overwrite to True if you want to ignore cached files

# Fetch videos and split videos into frames
image_folders = videointerface.main(
    videos_path, images_path, querie_path, "coordinates", "47.371667, 8.542222", max_results=1, num_vids=1, format="wv", duration='00:03:00', overwrite=False)

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
