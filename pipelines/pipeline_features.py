from pathlib import Path
import os
from cityslam.mapping import extract

from filelock import FileLock

# fix file permission problems
os.umask(0o002)

# setup necessary paths
base_dir = Path('/cluster/project/infk/courses/252-0579-00L/group07')

images_path = base_dir / 'datasets' / 'images'
output_path = base_dir / 'outputs' / 'model-features'


overwrite = False
# video_ids = ['gTHMvU3XHBk', 'TZIHy1cZJ-U']
video_ids = [p.name for p in list(images_path.iterdir())]

for video_id in video_ids:
    lock_path = output_path / f"{video_id}.lock"
    lock = FileLock(lock_path)
    with lock:
        extract.main(images_path, output_path, video_id, overwrite)
