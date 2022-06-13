from pathlib import Path
import argparse

import h5py
from filelock import FileLock


def copy_part(single_feature_file, common_feature_file, model_name, overwrite=False):

    assert Path(single_feature_file).exists(), single_feature_file

    lock_path = common_feature_file.parent / f"{common_feature_file.name}.lock"
    lock = FileLock(lock_path)
    with lock:
        with h5py.File(common_feature_file, 'a') as common_f:
            with h5py.File(single_feature_file, 'r') as single_f:
                if model_name not in list(single_f.keys()):
                    raise KeyError(f"model not found! {model_name}")

                if overwrite and model_name in list(common_f.keys()):
                    del common_f[model_name]

                if model_name not in list(common_f.keys()):
                    common_f[model_name] = h5py.ExternalLink(
                        single_feature_file, model_name)


"/cluster/home/tforgaard/Projects/zuricityslam/base/outputs/models-features/73IVzh0R-Lo"
def merge_match_files(model_path):

    match_files = Path(model_path).glob(
        "**/feats-superpoint-n4096-r1024_matches-superglue_pairs-sequential6-retrieval-netvlad6.h5")

    common_feature_file = Path(
        model_path) / "feats-superpoint-n4096-r1024_matches-superglue_pairs-sequential6-retrieval-netvlad6.h5"

    with h5py.File(common_feature_file, 'a') as common_f:
        for match_file in match_files:
            print(match_file)
            if match_file != common_feature_file:
                with h5py.File(str(match_file), 'r') as fd:
                    def visit_fn(_, obj):
                        if isinstance(obj, h5py.Dataset):
                            if not common_f.get(obj.name):
                                fd.copy(fd[obj.name], common_f, obj.name)
                    fd.visititems(visit_fn)


def update_features(feature_file, output, overwrite=False):

    model_name = feature_file.parent.name
    if "_part" in model_name:
        model_name = model_name.split("_part")[0]

    output_feature_path = Path(output) / feature_file.name

    copy_part(feature_file, output_feature_path, model_name, overwrite)

    return output_feature_path


def create_joint_feature_file(output, feature_list):

    for feature_file in feature_list:
        joint_path = update_features(feature_file, output, overwrite=True)

    return joint_path
