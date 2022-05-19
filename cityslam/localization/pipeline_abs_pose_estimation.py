#!/usr/bin/env python

from pathlib import Path
from helper_functions import get_images_from_recon, create_query_file, RANSAC_Transformation, model_path_2_name, model_name_2_path
import argparse

from hloc import extract_features, match_features
from hloc import pairs_from_retrieval, localize_sfm, visualization
from hloc.utils import viz_3d
import pycolmap

def main(images, models, output_dir, num_loc, N, reference, target, max_it, scale_std, max_distance_error, max_angle_error, min_inliers):
    
    # Setup the paths
    """
    target_path = model_name_2_path(target)
    reference_path = model_name_2_path(reference)

    target_name = model_path_2_name(target)
    reference_name = model_path_2_name(reference)
    """
    target_name = target
    reference_name = reference
    
    outputs = output_dir / 'merge' / f'merge_{target_name}_{reference_name}'
    outputs.mkdir(exist_ok=True, parents=True)

    # This is the reference and target model path
    reference_sfm = models / reference / 'sfm_sp+sg'  
    target_sfm = models / target / 'sfm_sp+sg'        

    # Retrieval pairs from target to reference
    loc_pairs = outputs / f'pairs-query-netvlad{num_loc}.txt'  # top-k retrieved by NetVLAD

    # Results file containing the estimated poses of the queries
    results = outputs / f'Merge_hloc_superpoint+superglue_netvlad{num_loc}.txt'

    # Configurations
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    # Path to local features
    #features = extract_features.main(feature_conf, images, outputs)
    features = models / 'feats-superpoint-n4096-r1024.h5'
    
    # Path to the global descriptors
    #global_descriptors = extract_features.main(retrieval_conf, images, outputs)
    global_descriptors = models / 'global-feats-netvlad.h5'
    
    # Get images from target model
    query_images = get_images_from_recon(target_sfm)

    # Use a subset of the images from the target model as queries
    query_list = query_images[::N]
    
    # Create a text file containing query images names and camera parameters
    queries_file = outputs / f'{target_name}_queries_with_intrinsics.txt'
    create_query_file(target_sfm, query_list, queries_file)
    
    # Do retrieval from target-queries to reference 
    pairs_from_retrieval.main(
        global_descriptors, loc_pairs, num_loc,
        query_list=query_list, db_model=reference_sfm, min_score=0.15)
    
    # Output path of matches
    matches = Path(models, f'{features.stem}_{matcher_conf["output"]}_{loc_pairs.stem}.h5')

    # Do matching for the pairs
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, features=features, matches=matches)

    # Do localization of queries
    localize_sfm.main(
        reference_sfm,
        queries_file,
        loc_pairs,
        features,
        loc_matches,
        results,
        #ransac_thresh=12,
        covisibility_clustering=False)  
    
    best_transform = RANSAC_Transformation(results, target_sfm, target, max_it, scale_std, max_distance_error, max_angle_error, min_inliers)    

    # TODO: safe best transform
    print(best_transform)
    
    reference_model = pycolmap.Reconstruction(reference_sfm)
    target_model_transformed = pycolmap.Reconstruction(target_sfm)
    target_model_transformed.transform(best_transform)

    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(fig, reference_model, color='rgba(255,0,0,0.2)', name="reference")
    viz_3d.plot_reconstruction(fig, target_model_transformed, color='rgba(0,255,0,0.2)', name="target transformed")

    fig.write_html(f'{outputs}/reconstruction_{target_name}_{reference_name}.html')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images',
                        help='Path to the dataset, default: %(default)s')
    """
    parser.add_argument('--models', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models-merge-testing-sent/model-matches-testing',
                        help='Path to the model directory, default: %(default)s')  
    """
    parser.add_argument('--models', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models-merge-testing-sent/models-merge-testing',
                        help='Path to the model directory, default: %(default)s')
    parser.add_argument('--output_dir', type=Path, default='/cluster/home/skalanan/',
                        help='Path to the output directory, default: %(default)s')                   
    parser.add_argument('--num_loc', type=int, default=10,
                        help='Number of retrieval pairs to generate for each query image: %(default)s')
    parser.add_argument('--N', type=int, default=5,
                        help='Use every Nth image from the images in the target reconstruction as query image: %(default)s')
    """                    
    parser.add_argument('--reference', type=str, default='TZIHy1cZJ-U/part0',
                        help='video id for reference model, %(default)s')
    parser.add_argument('--target', type=str, default='gTHMvU3XHBk/part0',
                        help='video id for target model, %(default)s')
    """
    parser.add_argument('--reference', type=str, default='TZIHy1cZJ-U',
                        help='video id for reference model, %(default)s')
    parser.add_argument('--target', type=str, default='gTHMvU3XHBk',
                        help='video id for target model, %(default)s')
    parser.add_argument('--max_it', type=int, default=200,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--scale_std', type=float, default=0.15306122448979592,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--max_distance_error', type=int, default=3,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--max_angle_error', type=int, default=5,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--min_inliers', type=int, default=100,
                        help='Min matching inliers needed to be part of RANSAC: %(default)s')              
    args = parser.parse_args()
    
    # Run mapping
    model = main(**args.__dict__)
