from pathlib import Path
import os
import subprocess
import tempfile
from time import sleep
from filelock import FileLock
from natsort import natsorted


# fix file permission problems
os.umask(0o002)

# setup necessary paths
base_dir = Path('/cluster/project/infk/courses/252-0579-00L/group07')

images_path = base_dir / 'datasets' / 'images'
image_splits = base_dir / 'datasets' / 'image_splits_new2'
output_path = base_dir / 'outputs' / 'models-features'

# Find all scenes
scene_ids = [str(p.relative_to(image_splits)).split("_images")[0] for p in natsorted(list(image_splits.glob("**/*_images.txt")))]
print(f"Total scenes: {len(scene_ids)}")

# Filter out the ones that are already done
scene_ids = [scene_id for scene_id in scene_ids if next((output_path / scene_id).glob("**/images.bin"), None) is None]
print(f"Scenes left: {len(scene_ids)}")

# Filter out the ones that are not ready (waiting for matching)
scene_ids = [scene_id for scene_id in scene_ids if next((output_path / scene_id).glob("**/database.db"), None) is not None]
print(f"Scenes ready for recon: {len(scene_ids)}")

P = min(8, len(scene_ids))  # NUMBER OF PARALLEL RECONSTRUCTIONS TO RUN

indexes = [i for i in range(P)]
processes = [None]*P
ready = [True]*P

while min(indexes) < len(scene_ids):
    for i, ind in enumerate(indexes):
        if ready[i] and ind < len(scene_ids):
            scene_id = scene_ids[ind]
            model_path = output_path / scene_id

            lock_path = output_path / f"{scene_id}.lock"
            lock = FileLock(lock_path)
            if lock.is_locked:
                indexes[i] += P
                processes[i] = None
                continue

            database = model_path / 'database.db'

            print(f"starting reconstruction for {scene_id}")
            f = tempfile.TemporaryFile()
            p = subprocess.Popen(
                ['python3', '-m', 
                'cityslam.mapping.reconstruction_subroutine', 
                '--sfm_dir', str(model_path), 
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
