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
output_path = base_dir / 'outputs' / 'models-features'

scene_ids = [str(p.relative_to(image_splits)).split("_images")[0] for p in sorted(list(image_splits.glob("**/*_images.txt")))]
# scene_ids = ['lN4j2iiFpgQ_part2']# ['_NmYvuEILw4_part2'] #, '_jJGc4r1mzk_part1']
# print(scene_ids)

for scene_id in scene_ids:

    image_list_path = Path(image_splits) / f"{scene_id}_images.txt"

    lock_path = output_path / f"{scene_id}.lock"
    lock = FileLock(lock_path, timeout=5)
    with lock:
        reconstruction = single_video_pipeline.main(
            images_path, image_list_path, output_path, video_id=scene_id, window_size=6, num_loc=6, pairing='sequential+retrieval', run_reconstruction=False)
