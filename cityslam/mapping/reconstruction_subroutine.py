
from filelock import Timeout, FileLock

from hloc.reconstruction import run_reconstruction

def main(sfm_dir, database, images_path, lock_path):
    lock = FileLock(lock_path, timeout=5)
    with lock:
        run_reconstruction(sfm_dir, database, images_path)
