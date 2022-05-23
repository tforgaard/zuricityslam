import json
import argparse
from pathlib import Path
from cityslam.localization import pipeline_abs_pose_estimation
from cityslam.utils.parsers import get_images_from_recon, model_path_2_name, model_name_2_path, get_model_base
from cityslam import logger

def main(scores_file, models, output_dir, num_loc, N, max_it, scale_std, min_model_score, max_distance_error, max_angle_error, only_sequential=False, overwrite=False,visualize=False):

    with open(scores_file) as f:
        score_dict = json.load(f)

    score_items = score_dict.items()
    tuple_pair = []
    for item in score_items:
        targ = (item[0],)
        sub_dict = item[1]
        tuple_pair += [targ+dict for dict in sub_dict.items()]

    sorted_scores = sorted(tuple_pair, key=lambda item: item[2], reverse=True)

    for (target_name, reference_name, score) in sorted_scores:
        if(score < min_model_score):
            continue

        reference = model_name_2_path(reference_name)
        target = model_name_2_path(target_name)

        if only_sequential:
            seq_n_target = int(target.parts[1].split("part")[-1])
            seq_n_ref = int(reference.parts[1].split("part")[-1])

            if not (seq_n_target + 1 == seq_n_ref or seq_n_target - 1 == seq_n_ref):
                continue

        logger.info(f"trying to merge target: {target} and reference {reference}")
        pipeline_abs_pose_estimation.main(models, output_dir, num_loc, N, reference,
                                          target, max_it, scale_std, max_distance_error, max_angle_error, overwrite=overwrite, visualize=visualize)

    """
    TODO:
    - check if the keys of score.json is the reference or target 
    - input correct file format with part0 stuff
    - Do we want to iterate over each pair and get partial reconstructions (& do this multiple times) 
        or do we want to merge the first pairs and update the list & use this model to build a bigger map? (instead of doing this multiple times)?
    - when creating the score.json file ignore pairs with itself?
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores_file', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/model-pairs/model_match_scores.json',
                        help='Path to the scores file, default: %(default)s')
    parser.add_argument('--models', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models',
                        help='Path to the model directory, default: %(default)s')
    parser.add_argument('--output_dir', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/model-pairs',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=10,
                        help='Number of retrieval pairs to generate for each query image: %(default)s')
    parser.add_argument('--N', type=int, default=5,
                        help='Use every Nth image from the images in the target reconstruction as query image: %(default)s')
    parser.add_argument('--min_model_score', type=float, default=0.8,
                        help='Min model match score: %(default)s')
    parser.add_argument('--max_it', type=int, default=1000,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--scale_std', type=float, default=0.1,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--max_distance_error', type=int, default=3,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--max_angle_error', type=int, default=5,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--only_sequential', action="store_true")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()

    # Run mapping
    model = main(**args.__dict__)
