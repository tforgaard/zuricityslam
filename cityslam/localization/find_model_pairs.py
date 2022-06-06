import argparse
from pathlib import Path
import numpy as np
import json

from hloc import pairs_from_retrieval
from hloc.utils.parsers import parse_retrieval

from .. import logger
from cityslam.utils.parsers import get_images_from_recon, model_path_2_name, model_name_2_path, get_model_base

def main(models_dir, outputs, num_loc, retrieval_interval, min_score, models_mask=None, overwrite=False, visualize=False):

    outputs = Path(outputs)
    if not outputs.exists():
        outputs.mkdir(mode=0o777, parents=True, exist_ok=True)

    # Recursively search for all models
    model_folders = [p.parent for p in Path(models_dir).glob("**/images.bin")]

    remove_folders = []
    for model_folder in model_folders:
        
        # If we have reconstructions in the PATH/models/[0-9] folders
        # Then we should remove the reconstruction in PATH, as this reconstruction
        # is the same as one of the ones in PATH/models/[0-9]
        if model_folder.name.isdigit():    
            rem_folder = model_folder.relative_to(models_dir).parent.parent
            remove_folders.append(rem_folder)

    # Make the model_folder paths relative to models_dir and remove redundant folders
    model_folders = [model_folder.relative_to(models_dir) for model_folder in model_folders if model_folder not in remove_folders]

    print(f"Scenes ready for pairing: {len(model_folders)}")

    # Optionally only include specific models
    if models_mask is not None:
        if isinstance(models_mask, str):
            models_mask = [models_mask]
        model_folders = [model_folder for model_folder in model_folders for model_mask in models_mask if model_mask in model_folder.parts]

    K = len(model_folders)
    logger.info(f"found {K} models")

    scores = np.zeros((K, K))

    models_dict = {m : model_m for m, model_m in enumerate(model_folders)}

    # global_descriptors = next(models_dir.glob("global-feats*.h5"))

    for n_ref, model_ref in enumerate(model_folders):

        img_names_ref = get_images_from_recon(Path(models_dir) / model_ref)
        descriptor_ref = next(get_model_base(models_dir, model_ref).glob("global-feats*.h5"))
    
        for n_target, model_target in enumerate(model_folders):
            if n_ref != n_target:
                
                # sfm_pairs = outputs / f'pairs-merge_{model_path_2_name(models_dict[n_target])}_{model_path_2_name(models_dict[n_ref])}_{num_loc}.txt'
                sfm_pairs = outputs / models_dict[n_target] / models_dict[n_ref] / f'pairs-merge-{num_loc}.txt'
                sfm_pairs.parent.mkdir(exist_ok=True, parents=True)

                img_names_target = get_images_from_recon(Path(models_dir) / model_target)
                descriptor_target = next(get_model_base(models_dir, model_target).glob("global-feats*.h5"))
                query = img_names_target[::retrieval_interval]
                
                if not sfm_pairs.exists() or overwrite:
                    
                    pairs = []
                    # Check to see if models are sequential partitions
                    if get_model_base(models_dir, model_target) == get_model_base(models_dir, model_ref):
                        pairs = check_for_common_images(img_names_target, img_names_ref, model_target, model_ref)

                    # Mask out already matched pairs
                    match_mask = np.zeros((len(query), len(img_names_ref)),dtype=bool)
                    for (p1, p2) in pairs:
                        if p1 in query:
                            match_mask[query.index(p1), img_names_ref.index(p2)] = True

                    # Find retrieval pairs
                    pairs_from_retrieval.main(descriptor_target, sfm_pairs,
                                            num_matched=num_loc, query_list=query, db_model=Path(models_dir) / model_ref, db_descriptors=descriptor_ref, min_score=min_score, match_mask=match_mask)

                    # Add common image pairs
                    retrieval = parse_retrieval(sfm_pairs)

                    for key, val in retrieval.items():
                        for match in val:
                            if (key, match) not in pairs:
                                pairs.append((key, match))

                    with open(sfm_pairs, 'w') as f:
                        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
                
                else:
                    logger.info("pairs already found, skipping retrieval...")

                    retrieval = parse_retrieval(sfm_pairs)

                # TODO make a better scoring alg, maybe sum up all the scores? and drop min_score
                scores[n_target, n_ref] = len(retrieval) / len(query)

    scores_file = outputs / "model_match_scores.json"
    
    scores_dict = {}

    # Load previous score file it it exists
    if scores_file.exists():
        with open(scores_file, "r") as f:
            scores_dict = json.load(f)

    if not scores_file.exists():
        scores_file.parent.mkdir(parents=True, exist_ok=True)

    for i in range(scores.shape[0]):
        if overwrite or scores_dict.get(model_path_2_name(models_dict[i])) is None: 
            scores_dict[model_path_2_name(models_dict[i])] = {}

        for j in range(scores.shape[1]):
            if overwrite or scores_dict[model_path_2_name(models_dict[i])].get(model_path_2_name(models_dict[j])) is None:
                scores_dict[model_path_2_name(models_dict[i])][model_path_2_name(models_dict[j])] = scores[i,j]
    
    # Cache score file
    with open(scores_file, "w+") as f:
        print(f"writing to file {scores_file}")
        json.dump(scores_dict, f)


    total_scores = np.zeros_like(scores)
    for j in range(scores.shape[0]):
        for i in range(j, scores.shape[1]):
            total_scores[i,j] = (scores[j,i] + scores[i,j]) / 2

    print(scores)
    print(total_scores)

    if visualize:
        import matplotlib.pyplot as plt
        plt.imshow(scores, interpolation='none')
        plt.colorbar()
        plt.savefig(outputs / "model_match_scores.png")


def check_for_common_images(img_names_target, img_names_ref, target, reference):

    pairs = []

    seq_n_target = int(target.parts[1].split("part")[-1])
    seq_n_ref = int(reference.parts[1].split("part")[-1])

    if seq_n_target + 1 == seq_n_ref or seq_n_target - 1 == seq_n_ref:
                    # look for same images in the two reconstructions..
        img_stems_target = set([img_name_target.split("/")[-1] for img_name_target in img_names_target])
        img_stems_ref = set([img_name_ref.split("/")[-1] for img_name_ref in img_names_ref])

        img_stems_common = img_stems_target.intersection(img_stems_ref)

        for img_stem_common in img_stems_common:

            pairs.append((  "/".join([target.parts[0], img_stem_common]), 
                            "/".join([reference.parts[0], img_stem_common])))

        logger.info(f"found {len(pairs)} pairs from common images")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models',
                        help='Path to the models, searched recursively, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/model-pairs',
                        help='Output path, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=7,
                        help='Number of image pairs for retrieval, default: %(default)s')
    parser.add_argument('--retrieval_interval', type=int, default=5,
                        help='How often to trigger retrieval: %(default)s')
    parser.add_argument('--min_score', type=float, default=0.15,
                        help='Minimum score for retrieval: %(default)s')
    parser.add_argument('--models_mask', nargs="+", default=None,
                        help='Only include given models: %(default)s')
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()

    main(**args.__dict__)
