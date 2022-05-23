#!/usr/bin/env python

from pathlib import Path
import os.path
from os.path import exists
from helper_functions import *
import argparse
import numpy as np
from hloc import extract_features, match_features
from hloc import pairs_from_retrieval, localize_sfm, visualization
from hloc.utils import viz_3d
import pycolmap

def RANSAC_Transformation(results, target_sfm, target, max_it, scale_std, max_distance_error, max_angle_error, min_inliers_estimates, min_inliers_transformations):
    max_distance_error = 0.5
    max_angle_error = 5 # in degrees
    max_inliers = 0
    best_transform = None
    best_query = ""
    best_scale = 1.0
    best_distance_error = 10000
    best_angle_error = 180
    max_it = 800
    num_it = 0

    scale_std = 0.3 / 1.96
    min_inliers_estimates=100
    min_inliers_transformations=10


    # TODO: load the pkl file to get number of inliers and ratio
    # We should trust pose estimates with a high number of inliers more than other ones...

    # Load the estimates for the query poses in the frame of the reference model
    pose_estimates = parse_pose_estimates(results)
    pose_estimates = filter_pose_estimates(pose_estimates, results, min_inliers_estimates)
    target_model = pycolmap.Reconstruction(target_sfm)

    # TODO: exchange outer loop with a while loop with max iterations and random sample
    # for img_name1, pose_est1 in pose_estimates.items():
    while num_it < max_it:
        ind = np.random.randint(len(pose_estimates.keys()))
        img_name1 = list(pose_estimates.keys())[ind]
        pose_est1 = pose_estimates[img_name1]
        # pose_est1 is the pose of the query in the frame of the reference model

        # Get a random scale sample
        scale1 = np.random.normal(1.0, scale_std)

        # We try to rotate the target model such that it aligns with the reference model
        target_model_trans_tmp = pycolmap.Reconstruction(target_sfm)

        # Pose of the query in the original target model frame
        pose_in_target1 = target_model.find_image_with_name(f'{target.split("/")[0]}/' + img_name1)

        # Transform which hopefully aligns the target model with the reference model
        transform1 = calculate_transform(pose_in_target1, pose_est1, scale1)
        target_model_trans_tmp.transform(transform1)


        inliers = 0
        inliers_d = []
        inliers_ang = []
        # 'Inner loop' comparing the transformation found against the other pose estimates
        for img_name2, pose_est2 in pose_estimates.items():

            pose_in_target2 = target_model_trans_tmp.find_image_with_name(f'{target.split("/")[0]}/' + img_name2)
            
            # This transform should be equal to a rotation matrix like identity 
            # and zero translation if the pose_est1 and pose_est2 agree 
            transform2 = calculate_transform(pose_in_target2, pose_est2)

            # Somebody needs to checkthis formula, supposed to compute the angle of the rotation
            theta = np.arccos((np.trace(pycolmap.qvec_to_rotmat(transform2.rotation)) - 1) / 2)
            angle = theta * 180 / np.pi

            # Do not know what scale we have...
            d = np.linalg.norm(transform2.translation)

            # Debug
            # print(f"query1: {img_name1}, query2: {img_name2},  angle error: {angle:.3f}, distance error: {d:.3f}")

            if d < max_distance_error and angle < max_angle_error:
                inliers+=1
                inliers_d.append(d)
                inliers_ang.append(angle)

        if inliers > max_inliers or (inliers == max_inliers and np.mean(inliers_d) < best_distance_error and np.mean(inliers_ang) < best_angle_error):
            max_inliers = inliers
            best_transform = transform1
            best_query = img_name1
            best_scale = scale1
            best_distance_error = np.mean(inliers_d)
            best_angle_error = np.mean(inliers_ang)
            print(f"currently best query: {best_query}, scale: {scale1}, {max_inliers}/{len(pose_estimates.keys())} inliers")
        
        num_it += 1

    if max_inliers < min_inliers_transformations:
        return None
    
    return best_transform

def main(images, models, output_dir, num_loc, N, reference, target, max_it, scale_std, max_distance_error, max_angle_error, min_inliers_estimates, min_inliers_transform):
    """
    # Setup the paths
    target_path = model_name_2_path(target)
    reference_path = model_name_2_path(reference)

    target_name = model_path_2_name(target)
    reference_name = model_path_2_name(reference)
    """
    outputs = output_dir / f'merge_{target.replace("/", "_")}_{reference.replace("/", "_")}'
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
    queries_file = outputs / f'{target.replace("/", "_")}_queries_with_intrinsics.txt'
    create_query_file(target_sfm, query_list, queries_file)
    
    # Do retrieval from target-queries to reference 
    if (not exists(loc_pairs)):
        pairs_from_retrieval.main(
            global_descriptors, loc_pairs, num_loc,
            query_list=query_list, db_model=reference_sfm, min_score=0.15)
    
    # Output path of matches
    matches = Path(models, f'{features.stem}_{matcher_conf["output"]}_{loc_pairs.stem}.h5')

    # Do matching for the pairs
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True)

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
    
    best_transform = RANSAC_Transformation(results, target_sfm, target, max_it, scale_std, max_distance_error, max_angle_error, min_inliers_estimates, min_inliers_transform)    
    
    if best_transform is not None:
        transform = best_transform.matrix
        txt = outputs / f'trans__{target.replace("/", "__")}__{reference.replace("/", "__")}.txt'
        np.savetxt(txt, transform[:-1, :], delimiter=',')
        
        reference_model = pycolmap.Reconstruction(reference_sfm)
        target_model_transformed = pycolmap.Reconstruction(target_sfm)
        target_model_transformed.transform(best_transform)

        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(fig, reference_model, color='rgba(255,0,0,0.2)', name="reference")
        viz_3d.plot_reconstruction(fig, target_model_transformed, color='rgba(0,255,0,0.2)', name="target transformed")

        fig.write_html(f'{outputs}/reconstruction_{target.replace("/", "_")}_{reference.replace("/", "_")}.html')
    
    else:
        print("could not find a transform")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--models', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models-features',
                        help='Path to the model directory, default: %(default)s')
    parser.add_argument('--output_dir', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/test/merge',
                        help='Path to the output directory, default: %(default)s')                   
    parser.add_argument('--num_loc', type=int, default=4,
                        help='Number of retrieval pairs to generate for each query image: %(default)s')
    parser.add_argument('--N', type=int, default=10,
                        help='Use every Nth image from the images in the target reconstruction as query image: %(default)s')              
    parser.add_argument('--reference', type=str, default='73IVzh0R-Lo/part1',
                        help='video id for reference model, %(default)s')
    parser.add_argument('--target', type=str, default='73IVzh0R-Lo/part0',
                        help='video id for target model, %(default)s')
    parser.add_argument('--max_it', type=int, default=400,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--scale_std', type=float, default=0.15306122448979592,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--max_distance_error', type=int, default=0.5,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--max_angle_error', type=int, default=5,
                        help='Max iteration for RANSAC: %(default)s')
    parser.add_argument('--min_inliers_estimates', type=int, default=100,
                        help='Min matching inliers needed to be part of RANSAC: %(default)s')
    parser.add_argument('--min_inliers_transform', type=int, default=10,
                        help='Min inliers needed in inner loop to be considered valid transformation: %(default)s')               
    args = parser.parse_args()
    
    # Run mapping
    model = main(**args.__dict__)
