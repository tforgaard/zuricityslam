from pathlib import Path
import os
import subprocess
import tempfile
from time import sleep

from hloc.reconstruction import run_reconstruction

# fix file permission problems
os.umask(0o002)

# setup necessary paths
base_dir = Path('/cluster/project/infk/courses/252-0579-00L/group07')

images_path = base_dir / 'datasets' / 'images'
image_splits = base_dir / 'datasets' / 'image_splits'
output_path = base_dir / 'outputs' / 'models-split-testing'

# scene_ids = ['_jJGc4r1mzk_part0']
scene_ids = [p.name.split("_images")[0] for p in sorted(list(image_splits.iterdir()))][:8]

P=4 # NUMBER OF PARALLEL RECONSTRUCTIONS TO RUN

indexes = [i for i in range(P)]
processes = [None]*P
ready = [False]*P

while max(indexes) < len(scene_ids) and sum(ready) == P:
    for i, ind in enumerate(indexes):
        if processes[i] is not None:
            if p.poll() is not None:
                f.seek(0)
                print(f.read())
                f.close()
                ready[i] = True
                indexes[i] += P
        else:
            ready[i] = True      

        if ready[i] and ind < len(scene_ids):
            scene_id = scene_ids[ind]
            output_model = output_path / scene_id
            sfm_dir = output_model / 'sfm_sp+sg'
            database = sfm_dir / 'database.db'

            f = tempfile.TemporaryFile()
            p = subprocess.Popen(['python3', '-c', f"from hloc.reconstruction import run_reconstruction; run_reconstruction({sfm_dir}, {database}, {images_path})"], stdout=f)

            processes[i] = (p, f)
            ready[i] = False

    sleep(5)
