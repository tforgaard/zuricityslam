

from pathlib import Path
import argparse
from cityslam.mapping import update_features


def main(models_path, overwrite=False):
    for model_dir in Path(models_path).iterdir():
        if model_dir.is_dir():
            
            feature_file = next(model_dir.glob("feats*.h5"), None)
            retrieval_file = next(model_dir.glob("global-feats*.h5"), None)

            if feature_file is not None:
                update_features.main(feature_file, models_path, overwrite)

            if retrieval_file is not None:
                update_features.main(retrieval_file, models_path, overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_path', type=Path, default='/cluster/home/tforgaard/Projects/zuricityslam/base/outputs/model-matches-testing')
    parser.add_argument('--overwrite', action="store_true")
    args = parser.parse_args()

    main(**args.__dict__)
