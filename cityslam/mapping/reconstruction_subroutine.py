
from filelock import FileLock
import argparse
from pathlib import Path
import os

from hloc.reconstruction import run_reconstruction

def main(sfm_dir, database, images_path, lock_path):
    os.umask(0o002)
    lock = FileLock(lock_path)
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
