
from filelock import Timeout, FileLock
import argparse
from pathlib import Path

from hloc.reconstruction import run_reconstruction

def main(sfm_dir, database, images_path, lock_path):
    lock = FileLock(lock_path, timeout=5)
    with lock:
        run_reconstruction(sfm_dir, database, images_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sfm_dir', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--images_path', type=Path, required=True)
    parser.add_argument('--lock_path', type=Path, required=True)
    args = parser.parse_args()

    main(**args.__dict__)
