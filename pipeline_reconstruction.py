from pathlib import Path
import os
import subprocess
import tempfile
from time import sleep
from filelock import Timeout, FileLock


# fix file permission problems
os.umask(0o002)

# setup necessary paths
base_dir = Path('/cluster/project/infk/courses/252-0579-00L/group07')

images_path = base_dir / 'datasets' / 'images'
image_splits = base_dir / 'datasets' / 'image_splits'
output_path = base_dir / 'outputs' / 'models-features'


scene_ids = [str(p.relative_to(image_splits)).split("_images")[0] for p in sorted(list(image_splits.glob("**/*_images.txt")))]

P = min(4, len(scene_ids))  # NUMBER OF PARALLEL RECONSTRUCTIONS TO RUN

indexes = [i for i in range(P)]
processes = [None]*P
ready = [True]*P

while min(indexes) < len(scene_ids):
    for i, ind in enumerate(indexes):
        if ready[i] and ind < len(scene_ids):
            scene_id = scene_ids[ind]
            model_path = output_path / scene_id

            lock_path = output_path / f"{scene_id}.lock"
            lock = FileLock(lock_path, timeout=5)
            if lock.is_locked:
                indexes[i] += P
                processes[i] = None
                continue

            sfm_dir = model_path / 'sfm_sp+sg'
            database = sfm_dir / 'database.db'

            f = tempfile.TemporaryFile()
            p = subprocess.Popen(
                ['python3', '-m', 
                'cityslam.mapping.reconstruction_subroutine', 
                '--sfm_dir', str(sfm_dir), 
                '--database', str(database), 
                '--images_path', str(images_path), 
                '--lock_path', str(lock_path)], stdout=f)

            processes[i] = (p, f)
            ready[i] = False

        if processes[i] is not None:
            (p, f) = processes[i]
            if p.poll() is not None:

                f.seek(0)
                print(f.read())
                f.close()

                ready[i] = True
                processes[i] = None
                indexes[i] += P

    sleep(5)
