import argparse
from pathlib import Path
from xmlrpc.client import Boolean
import numpy as np
import json

from hloc import pairs_from_retrieval

from hloc.reconstruction import import_matches, get_image_ids, create_empty_db
from hloc.triangulation import geometric_verification

from hloc.utils.parsers import parse_image_lists, parse_retrieval
from hloc.utils.read_write_model import read_images_binary
from hloc.utils.io import list_h5_names

from .. import logger

import pycolmap

def get_images_from_recon(sfm_model):
    """Get a sorted list of images in a reconstruction"""
    # NB! This will most likely be a SUBSET of all the images in a folder like images/gTHMvU3XHBk

    if isinstance(sfm_model, (str, Path)):
        sfm_model = pycolmap.Reconstruction(sfm_model)
    
    img_list = [img.name for img in sfm_model.images.values()]
    
    return sorted(img_list)


def main(models_dir, outputs, num_loc, retrieval_interval, min_score, overwrite=False):

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
            rem_folder = model_folder.relative_to(models_dir)
            remove_folders.append(rem_folder)

    # Make the model_folder paths relative to models_dir and remove redundant folders
    model_folders = [model_folder.relative_to(models_dir) for model_folder in model_folders if model_folder not in remove_folders]

    K = len(model_folders)
    logger.info(f"found {len(K)} models")
    
    scores = np.zeros((K, K))

    models_dict = {m : model_m for m, model_m in enumerate(model_folders)}

    global_descriptors = list(models_dir.glob("global-feats*.h5"))[0]

    for n_ref, model_ref in enumerate(model_folders):

        img_names_ref = get_images_from_recon(Path(models_dir) / model_ref)
        #img_names_ref = sorted([image.name for image in read_images_binary(model_ref / "sfm_sp+sg" / 'images.bin').values()])
    
        for n_target, model_target in enumerate(model_folders):
            if n_ref != n_target:
                
                img_names_target = get_images_from_recon(Path(models_dir) / model_target)
                # db_names_target = sorted([image.name for image in read_images_binary(model_target / "sfm_sp+sg" / 'images.bin').values()])
                # db_names_n = list_h5_names(descriptor_n)

                query = img_names_target[::retrieval_interval]

                # TODO: make this a file in a folder structure instead...
                sfm_pairs = outputs / f'pairs-merge_{models_dict[n_target]}_{models_dict[n_ref]}_{num_loc}.txt'


                # Check to see if models are sequential partitions
                video_id_target =  model_target.parts[0]
                video_id_ref =  model_ref.parts[0]
                pairs = check_for_common_images(img_names_target, img_names_ref, video_id_target, video_id_ref)

                # Mask out already matched pairs
                # already_matched_ref_imgs = [ref_img for (_, ref_img) in pairs]
                # db_list = [img_name_ref for img_name_ref in img_names_ref if img_name_ref not in already_matched_ref_imgs]
                match_mask = np.zeros((len(img_names_target, img_names_ref)),dtype=bool)
                for (p1, p2) in pairs:
                    match_mask[img_names_target.index(p1), img_names_ref.index(p2)] = True

                pairs_from_retrieval.main(global_descriptors, sfm_pairs,
                                          num_matched=num_loc, query_list=query, db_model=Path(models_dir) / model_ref, min_score=min_score, match_mask=match_mask)

                retrieval = parse_retrieval(sfm_pairs)

                for key, val in retrieval.items():
                    for match in val:
                        if (key, match) not in pairs:
                            pairs.append((key, match))

                with open(sfm_pairs, 'w') as f:
                    f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


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
        if overwrite or scores_dict.get(models_dict[i]) is None: 
            scores_dict[models_dict[i]] = {}

        for j in range(scores.shape[1]):
            if overwrite or scores_dict[models_dict[i]].get([models_dict[j]]) is None:
                scores_dict[models_dict[i]][models_dict[j]] = scores[i,j]
    
    # Cache score file
    with open(scores_file, "w+") as f:
        json.dump(scores_dict, f)


    total_scores = np.zeros_like(scores)
    for j in range(scores.shape[0]):
        for i in range(j, scores.shape[1]):
            total_scores[i,j] = (scores[j,i] + scores[i,j]) / 2

    print(scores)
    print(total_scores)

    # TODO we should do this for multiple pairs...
    # find best score from total scores
    # do same thing but switch target and reference
    # try to use model_merger...?
    # or do a reconstruction...?

    #best_pair = np.unravel_index(scores.argmax(), scores.shape)
    # best_pair = np.unravel_index(total_scores.argmax(), total_scores.shape)

    # target_ind, reference_ind = best_pair
    # target = models_dict[target_ind]
    # reference = models_dict[reference_ind]

    # features = list(models_dir.glob(f'feats-*.h5'))[0]


    # # output of merged models
    # merged_models = outputs / "merged"
    # if not merged_models.exists():
    #     merged_models.mkdir(mode=0o777, parents=True, exist_ok=True)
    
    # # pair files
    # merge_pairs = merged_models / "merge_pairs.txt"
    # merge_pairs_ref_target = list(outputs.glob(f"pairs-merge_{reference}_{target}*.txt"))[0]
    # merge_pairs_target_ref = list(outputs.glob(f"pairs-merge_{target}_{reference}*.txt"))[0]
    
    # find_unique_pairs(merge_pairs_ref_target, merge_pairs_target_ref, merge_pairs)

    # merging_matches_output = outputs / f"matches_{target}_{reference}.h5"


def check_for_common_images(img_names_target, img_names_ref, video_id_target, video_id_ref):

    pairs = []

    if "part" in video_id_target and "part" in video_id_ref:
        seq_n_target = int(video_id_target.split("part")[-1])
        seq_n_ref = int(video_id_ref.split("part")[-1])

        if seq_n_target + 1 == seq_n_ref or seq_n_target - 1 == seq_n_ref:
                        # look for same images in the two reconstructions..
            img_stems_target = set([img_name_target.split("/")[-1] for img_name_target in img_names_target])
            img_stems_ref = set([img_name_ref.split("/")[-1] for img_name_ref in img_names_ref])

            img_stems_common = img_stems_target.intersection(img_stems_ref)


            for img_stem_common in img_stems_common:

                pairs.append((  "/".join([video_id_target, img_stem_common]), 
                                "/".join([video_id_ref, img_stem_common])))

    return pairs

def find_unique_pairs(merge_pairs_ref_target, merge_pairs_target_ref, output):
    retrieval_ref_target = parse_retrieval(merge_pairs_ref_target)
    retrieval_target_ref = parse_retrieval(merge_pairs_target_ref)

    pairs_set = set()
    
    for key, val in retrieval_ref_target.items():
        for match in val:
            pairs_set.add(frozenset((key, match)))

    for key, val in retrieval_target_ref.items():
        for match in val:
            pairs_set.add(frozenset((key, match)))
    
    pairs = [tuple(pair) for pair in pairs_set]

    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/theo/outputs/models',
                        help='Path to the models, searched recursively, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/theo/outputs/merge',
                        help='Output path, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=7,
                        help='Number of image pairs for retrieval, default: %(default)s')
    parser.add_argument('--retrieval_interval', type=int, default=5,
                        help='How often to trigger retrieval: %(default)s')
    parser.add_argument('--min_score', type=float, default=0.5,
                        help='Minimum score for retrieval: %(default)s')
    args = parser.parse_args()

    main(**args.__dict__)
