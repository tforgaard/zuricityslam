from pathlib import Path
import os
import subprocess
import tempfile
from cityslam.mapping import extract

# fix file permission problems
os.umask(0o002)

# setup necessary paths
base_dir = Path('/cluster/project/infk/courses/252-0579-00L/group07')

images_path = base_dir / 'datasets' / 'images'
output_path = base_dir / 'outputs' / 'models-split-testing'

overwrite = False
# video_ids = ['gTHMvU3XHBk', 'TZIHy1cZJ-U']


# video_ids = [p.name.split("_images")[0] for p in sorted(list(image_splits.iterdir()))][:8]
video_ids = [p.name for p in list(images_path.iterdir())][:8]


for video_id in video_ids:
    extract.main(images_path, output_path, video_id)
