import argparse
from pathlib import Path
import numpy as np
import json

from hloc import pairs_from_retrieval_resampling
from hloc.utils.parsers import parse_retrieval

from cityslam import logger
from cityslam.utils.parsers import get_images_from_recon, model_path_2_name, model_name_2_path, get_model_base, find_models


def main(models_dir, outputs, num_loc, retrieval_interval, resample_runs, min_score, models_mask=None, overwrite=False, visualize=False):

    outputs = Path(outputs)
    if not outputs.exists():
        outputs.mkdir(mode=0o777, parents=True, exist_ok=True)

    model_folders = find_models(models_dir, models_mask)
    
    logger.info(f"Scenes ready for pairing: {len(model_folders)}")

    K = len(model_folders)
    
    model_to_ind = {model_path_2_name(
        k): i for i, k in enumerate(model_folders)}

    scores = np.zeros((K, K))
    scores_file = outputs / "model_match_scores.json"
    scores_dict = load_score_file(scores_file)

    load_scores(model_folders, model_to_ind, scores, scores_dict)

    models_dict = {m: model_m for m, model_m in enumerate(model_folders)}

    for n_ref, model_ref in enumerate(model_folders):

        img_names_ref = get_images_from_recon(Path(models_dir) / model_ref)
        descriptor_ref = next(get_model_base(
            models_dir, model_ref).glob("global-feats*.h5"))

        for n_target, model_target in enumerate(model_folders):
            if n_ref != n_target:

                logger.info(f"pairing model {model_target} with {model_ref}")

                sfm_pairs = outputs / models_dict[n_target] / models_dict[n_ref] / f'pairs-merge-{num_loc}.txt'
                sfm_pairs.parent.mkdir(exist_ok=True, parents=True)

                img_names_target = get_images_from_recon(
                    Path(models_dir) / model_target)
                descriptor_target = next(get_model_base(
                    models_dir, model_target).glob("global-feats*.h5"))
                queries = img_names_target

                if not sfm_pairs.exists() or overwrite:

                    pairs = []
                    # Check to see if models are sequential partitions
                    if get_model_base(models_dir, model_target) == get_model_base(models_dir, model_ref):
                        pairs = check_for_common_images(
                            img_names_target, img_names_ref, model_target, model_ref)

                    # Mask out already matched pairs
                    match_mask = np.zeros(
                        (len(queries), len(img_names_ref)), dtype=bool)
                    for (p1, p2) in pairs:
                        if p1 in queries:
                            match_mask[queries.index(
                                p1), img_names_ref.index(p2)] = True

                    # Find retrieval pairs
                    _, score = pairs_from_retrieval_resampling.main(descriptor_target, sfm_pairs,
                                                                    num_matched=num_loc, query_list=queries,
                                                                    db_model=Path(models_dir) / model_ref, db_descriptors=descriptor_ref,
                                                                    min_score=min_score, match_mask=match_mask,
                                                                    query_interval=retrieval_interval, resample_runs=resample_runs,
                                                                    visualize=False)

                    # Add common image pairs
                    retrieval = parse_retrieval(sfm_pairs)

                    for key, val in retrieval.items():
                        for match in val:
                            if (key, match) not in pairs:
                                pairs.append((key, match))

                    with open(sfm_pairs, 'w') as f:
                        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

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
                    logger.info("pairs already found, skipping retrieval...")

    # Load score file it it exists
    scores_dict = load_score_file(scores_file)
    load_scores(model_folders, model_to_ind, scores, scores_dict)

    total_scores = np.zeros_like(scores)
    for j in range(scores.shape[0]):
        for i in range(j, scores.shape[1]):
            total_scores[i, j] = (scores[j, i] + scores[i, j]) / 2

    print(scores)
    print(total_scores)

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


def save_score(scores_file, score, target, ref, overwrite):

    scores_dict = load_score_file(scores_file)

    target = model_path_2_name(target)
    ref = model_path_2_name(ref)

    if overwrite or scores_dict.get(target) is None:
        scores_dict[target] = {}

    if overwrite or scores_dict[target].get(ref) is None:
        scores_dict[target][ref] = score

    # Cache score file
    with open(scores_file, "w+") as f:
        print(f"writing to file {scores_file}")
        json.dump(scores_dict, f)


def check_for_common_images(img_names_target, img_names_ref, target, reference):

    pairs = []

    seq_n_target = int(target.parts[1].split("part")[-1])
    seq_n_ref = int(reference.parts[1].split("part")[-1])

    if seq_n_target + 1 == seq_n_ref or seq_n_target - 1 == seq_n_ref:
        # look for same images in the two reconstructions..
        img_stems_target = set([img_name_target.split("/")[-1]
                               for img_name_target in img_names_target])
        img_stems_ref = set([img_name_ref.split("/")[-1]
                            for img_name_ref in img_names_ref])

        img_stems_common = img_stems_target.intersection(img_stems_ref)

        for img_stem_common in img_stems_common:

            pairs.append(("/".join([target.parts[0], img_stem_common]),
                          "/".join([reference.parts[0], img_stem_common])))

        logger.info(f"found {len(pairs)} pairs from common images")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models',
                        help='Path to the models, searched recursively, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/model-pairs-test',
                        help='Output path, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=10,
                        help='Number of image pairs for retrieval, default: %(default)s')
    parser.add_argument('--retrieval_interval', type=int, default=15,
                        help='How often to trigger retrieval: %(default)s')
    parser.add_argument('--resample_runs', type=int, default=3,
                        help='How often to trigger retrieval: %(default)s')
    parser.add_argument('--min_score', type=float, default=0.1,
                        help='Minimum score for retrieval: %(default)s')
    parser.add_argument('--models_mask', nargs="+", default=None,
                        help='Only include given models: %(default)s')
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()

    args.visualize = True
    args.models_mask = ["DdxEr70jC6E"]
    main(**args.__dict__)
