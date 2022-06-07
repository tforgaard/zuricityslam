from pathlib import Path
import os
from filelock import FileLock
from natsort import natsorted

from cityslam.mapping import single_video_pipeline


# fix file permission problems
os.umask(0o002)

# setup necessary paths
base_dir = Path('/cluster/project/infk/courses/252-0579-00L/group07')

images_path = base_dir / 'datasets' / 'images'
image_splits = base_dir / 'datasets' / 'image_splits'
output_path = base_dir / 'outputs' / 'models-features'

# Find all scenes
scene_ids = [str(p.relative_to(image_splits)).split("_images")[0] for p in natsorted(list(image_splits.glob("**/*_images.txt")))]
print(f"Total scenes: {len(scene_ids)}")

# Filter out the ones that are already done
scene_ids = [scene_id for scene_id in scene_ids if next((output_path / scene_id).glob("**/database.db"), None) is None]
print(f"Scenes left: {len(scene_ids)}")

for scene_id in scene_ids:

    image_list_path = Path(image_splits) / f"{scene_id}_images.txt"

    lock_path = output_path / f"{scene_id}.lock"
    lock = FileLock(lock_path)
    with lock:
        reconstruction = single_video_pipeline.main(
            images_path, image_list_path, output_path, video_id=scene_id, window_size=6, num_loc=6, pairing='sequential+retrieval', run_reconstruction=False, overwrite=False)
