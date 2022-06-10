import argparse
from pathlib import Path
import numpy as np
import json


from cityslam import logger
from cityslam.utils.parsers import model_path_2_name, model_name_2_path, find_models
from cityslam.localization import model_pairs

def main(models, outputs, models_mask=None, overwrite=False, visualize=False):

    outputs = Path(outputs)
    if not outputs.exists():
        outputs.mkdir(mode=0o777, parents=True, exist_ok=True)

    model_folders = find_models(models, models_mask)
    
    logger.info(f"Scenes ready for pairing: {len(model_folders)}")

    K = len(model_folders)
    
    model_to_ind = {model_path_2_name(
        k): i for i, k in enumerate(model_folders)}

    scores = np.zeros((K, K))
    scores_file = outputs / "model_match_scores.json"
    scores_dict = load_score_file(scores_file)

    load_scores(model_folders, model_to_ind, scores, scores_dict)

    # models_dict = {m: model_m for m, model_m in enumerate(model_folders)}

    for n_target, model_target in enumerate(model_folders):

        for n_ref, model_ref in enumerate(model_folders):

            if n_ref >= n_target:
                continue

            if check_score(scores_file, model_target, model_ref) is None or overwrite:

                _, score = model_pairs.main(models, outputs, model_target, model_ref, overwrite=overwrite, visualize=False)

                if score is not None:

                    scores[n_target, n_ref] = score

                    if visualize:
                        import matplotlib.pyplot as plt
                        plt.clf()
                        plt.imshow(scores, interpolation='none')
                        plt.colorbar()
                        plt.savefig(outputs / "model_match_scores.png")

                    save_score(scores_file, score, model_target,
                                model_ref, overwrite)

            else:
                logger.info("score already found")

    # Load score file it it exists
    scores_dict = load_score_file(scores_file)
    load_scores(model_folders, model_to_ind, scores, scores_dict)
    print(scores)

    if visualize:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.imshow(scores, interpolation='none')
        plt.colorbar()
        plt.savefig(outputs / "model_match_scores.png")


def load_scores(model_folders, model_to_ind, scores, scores_dict):
    keys = set(list(scores_dict.keys()))
    more_keys = [list(v.keys()) for v in scores_dict.values()]

    for key in more_keys:
        keys.update(key)

    for k, v in scores_dict.items():
        for k2, score in v.items():
            if model_name_2_path(k) in model_folders and model_name_2_path(k2) in model_folders:
                scores[model_to_ind[k]][model_to_ind[k2]] = score


def load_score_file(scores_file):
    scores_dict = {}

    # Load previous score file it it exists
    if scores_file.exists():
        with open(scores_file, "r") as f:
            scores_dict = json.load(f)

    if not scores_file.exists():
        scores_file.parent.mkdir(parents=True, exist_ok=True)

    return scores_dict


def check_score(scores_file, target, ref):
    scores_dict = load_score_file(scores_file)

    target = model_path_2_name(target)
    ref = model_path_2_name(ref)

    if scores_dict.get(target) is None:
        return None

    return scores_dict[target].get(ref)


def save_score(scores_file, score, target, ref, overwrite):

    scores_dict = load_score_file(scores_file)

    target = model_path_2_name(target)
    ref = model_path_2_name(ref)

    if scores_dict.get(target) is None:
        scores_dict[target] = {}

    if overwrite or scores_dict[target].get(ref) is None:
        scores_dict[target][ref] = score

    # Cache score file
    with open(scores_file, "w+") as f:
        print(f"writing to file {scores_file}")
        json.dump(scores_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models-features',
                        help='Path to the models, searched recursively, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models-scores',
                        help='Output path, default: %(default)s')
    parser.add_argument('--models_mask', nargs="+", default=None,
                        help='Only include given models: %(default)s')
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()

    main(**args.__dict__)
