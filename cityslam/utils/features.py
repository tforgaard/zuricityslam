
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
            with h5py.File(single_feature_file,'r') as single_f:
                if model_name not in list(single_f.keys()):
                    print(f"model not found! {model_name}")
                    return

                if overwrite and model_name in list(common_f.keys()):
                    del common_f[model_name]

                if model_name not in list(common_f.keys()):
                    common_f[model_name] = h5py.ExternalLink(single_feature_file, model_name)


def update_features(feature_file, output, overwrite=False):
        
    model_name = feature_file.parent.name
    if "_part" in model_name:
        model_name = model_name.split("_part")[0]

    output_feature_path = Path(output) / feature_file.name
    
    copy_part(feature_file, output_feature_path, model_name, overwrite)

    return output_feature_path



def update_joint_features(models_path, output, models_list, overwrite=False):
    if models_list is None:
        models_list = [m.name for m in Path(models_path).iterdir()]
    for model_dir in Path(models_path).iterdir():
        if model_dir.is_dir() and model_dir.name in models_list:
            feature_file = next(model_dir.glob("feats*.h5"), None)
            retrieval_file = next(model_dir.glob("global-feats*.h5"), None)

            if feature_file is not None:
                update_features(feature_file, models_path, overwrite)

            if retrieval_file is not None:
                update_features(retrieval_file, models_path, overwrite)



def create_joint_feature_file(output, models_path, models_list, type="features"):
    
    if isinstance(models_list, str):
        models_list = [models_list]

    for model_dir in Path(models_path).iterdir():
        if model_dir.is_dir() and model_dir.name in models_list:

            if type == "features":
                feature_file = next(model_dir.glob("feats*.h5"), None)

                if feature_file is not None:
                    update_features(feature_file, output, overwrite=True)

            elif type == "descriptors":
                retrieval_file = next(model_dir.glob("global-feats*.h5"), None)


                if retrieval_file is not None:
                    update_features(retrieval_file, output, overwrite=True)

    return next(output.glob("feats*.h5"), None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default= '/cluster/home/tforgaard/Projects/zuricityslam/base/outputs/test/tmp',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--models_path', type=Path, default='/cluster/home/tforgaard/Projects/zuricityslam/base/outputs/models')
    parser.add_argument('--models_list', nargs="+", type=str, default=None)
    
    args = parser.parse_args()

    create_joint_feature_file(**args.__dict__)