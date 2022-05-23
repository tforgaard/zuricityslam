import json
import argparse
from pathlib import Path
import pipeline_abs_pose_estimation

def main(scores_file, images, models, output_dir, num_loc, N, reference, target, max_it, scale_std, max_distance_error, max_angle_error):
    
    with open(scores_file) as f:
        score_dict = json.load(f)

    score_items = score_dict.items()
    tuple_pair = []
    for item in score_items:
        reference = (item[0],)
        sub_dict = item[1]
        tuple_pair += [reference+dict for dict in sub_dict.items()]

    sorted_scores = sorted(tuple_pair, key=lambda item: item[2], reverse=True)
    print(sorted_scores)

    for (reference, target, score) in sorted_scores:
        if(score == 0.0): 
            continue
        
        # TODO fix this!!
        reference_name = str(reference).replace('__part0__sfm_sp+sg', '')
        target_name = str(target).replace('__part0__sfm_sp+sg', '')

        pipeline_abs_pose_estimation.main(images, models, output_dir, num_loc, N, reference_name, target_name, max_it, scale_std, max_distance_error, max_angle_error)

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
    parser.add_argument('--scores_file', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/theo/outputs/model-matches-testing/merge/model_match_scores.json',
                        help='Path to the scores file, default: %(default)s')
    parser.add_argument('--images', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--models', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models-merge-testing-sent/models-merge-testing',
                        help='Path to the model directory, default: %(default)s')   
    parser.add_argument('--output_dir', type=Path, default='/cluster/home/skalanan/',
                        help='Path to the output directory, default: %(default)s')                   
    parser.add_argument('--num_loc', type=int, default=10,
                        help='Number of retrieval pairs to generate for each query image: %(default)s')
    parser.add_argument('--N', type=int, default=5,
                        help='Use every Nth image from the images in the target reconstruction as query image: %(default)s')
    parser.add_argument('--reference', type=str, default='TZIHy1cZJ-U/part0',
                        help='video id for reference model, %(default)s')
    parser.add_argument('--target', type=str, default='gTHMvU3XHBk/part0',
                        help='video id for target model, %(default)s')
    parser.add_argument('--max_it', type=int, default=200,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--scale_std', type=float, default=0.15306122448979592,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--max_distance_error', type=int, default=3,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--max_angle_error', type=int, default=5,
                        help='Max iteration for RANSAC: %(default)s')                    
    args = parser.parse_args()
    
    # Run mapping
    model = main(**args.__dict__)
