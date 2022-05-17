
from pathlib import Path
import argparse

import h5py
from filelock import Timeout, FileLock

def copy_part(single_feature_file, common_feature_file, model_name, overwrite=False):

    assert Path(single_feature_file).exists(), single_feature_file

    lock_path = common_feature_file.parent / f"{common_feature_file.name}.lock"
    lock = FileLock(lock_path, timeout=5)
    with lock:
        with h5py.File(common_feature_file,'a') as common_f:
            # print(list(common_f.keys()))
            with h5py.File(single_feature_file,'r') as single_f:
                if model_name not in list(single_f.keys()):
                    print(f"model not found! {model_name}")
                    return

                if overwrite and model_name in list(common_f.keys()):
                    del common_f[model_name]

                if model_name not in list(common_f.keys()):
                    # common_f[model_name] = h5py.ExternalLink(single_feature_file, model_name)
                    single_f.copy(model_name, common_f)


def main(feature_file, output, overwrite=False):
        
    model_name = feature_file.parent.name
    if "_part" in model_name:
        model_name = model_name.split("_part")[0]

    output_feature_path = Path(output) / feature_file.name
    
    copy_part(feature_file, output_feature_path, model_name, overwrite)

    return output_feature_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=Path, default='/cluster/home/tforgaard/Projects/zuricityslam/base/outputs/models-merge-testing2/TZIHy1cZJ-U')
    parser.add_argument('--output', type=Path, default= '/cluster/home/tforgaard/Projects/zuricityslam/base/outputs/models-merge-testing2/',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--overwrite', action="store_true")
    
    args = parser.parse_args()

    main(**args.__dict__)