from pathlib import Path
import os
from filelock import Timeout, FileLock

from cityslam.mapping import single_video_pipeline


# fix file permission problems
os.umask(0o002)

# setup necessary paths
base_dir = Path('/cluster/project/infk/courses/252-0579-00L/group07')


images_path = base_dir / 'datasets' / 'images'
image_splits = base_dir / 'datasets' / 'image_splits'
output_path = base_dir / 'outputs' / 'reconstructions'

# scene_ids = ['_jJGc4r1mzk_part0']
scene_ids = [p.name.split("_images")[0] for p in sorted(list(image_splits.iterdir()))][:8]


for scene_id in scene_ids:
    lock_path = output_path / f"{scene_id}.lock"
    lock = FileLock(lock_path, timeout=5)
    with lock:
        reconstruction = single_video_pipeline.main(
            images_path, image_splits, output_path, video_id=scene_id, window_size=6, num_loc=6, pairing='sequential+retrieval', run_reconstruction=False)
