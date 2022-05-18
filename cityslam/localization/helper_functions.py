from pathlib import Path
import numpy as np

from hloc import extract_features, match_features
from hloc import pairs_from_retrieval, localize_sfm, visualization
from hloc.utils import viz_3d
import pycolmap

def get_images_from_recon(sfm_model):
    """Get a sorted list of images in a reconstruction"""
    # NB! This will most likely be a SUBSET of all the images in a folder like images/gTHMvU3XHBk

    if isinstance(sfm_model, (str, Path)):
        sfm_model = pycolmap.Reconstruction(sfm_model)
    
    img_list = [img.name for img in sfm_model.images.values()]
    
    return sorted(img_list)

def create_query_file(sfm_model, query_list, output):
    """Create a query file used for localization"""

    # Format:
    # img_name cam_model width height cam_params0, cam_params1,...

    if isinstance(sfm_model, (str, Path)):
        sfm_model = pycolmap.Reconstruction(sfm_model)

    cams = {}

    for k, cam in sfm_model.cameras.items():
        params = " ".join([f"{p:.6f}" for p in cam.params])
        cams[k] = f"{cam.model_name} {cam.width} {cam.height} {params}"

    with open(output, "w+") as file:
        for query in query_list:
            i = sfm_model.find_image_with_name(query)
            file.write(f"{query} {cams[i.camera_id]}\n")


def parse_pose_estimates(pose_estimate_file):
    pose_estimates = np.genfromtxt(pose_estimate_file, delimiter=' ', dtype=None, encoding=None)

    pose_dict = {}
    for pose_est in pose_estimates:
        img_name, *qvec, x, y, z = pose_est
        pose_dict[img_name] = pycolmap.Image(tvec=[x, y, z], qvec=np.array(qvec))
    
    return pose_dict


def calculate_transform(pose1, pose2, scale=1.0):
    """Calculates the transformation which aligns model 1 with model 2, pose{1,2} is the pose of the same image in the two different models"""
    
    # Rotation matrices which transforms from the local frame of the pose (image) to the respective world frames 1 and 2
    R_c_w1 = pycolmap.qvec_to_rotmat(pose1.qvec).T
    R_c_w2 = pycolmap.qvec_to_rotmat(pose2.qvec).T

    # Rotation matrix from world frame 1 to 2
    R_w1_w2 = R_c_w2 @ R_c_w1.T

    # Translation from respective world frame to the pose (image) center in the local frame
    r_w1c_c = - pose1.tvec
    r_w2c_c = - pose2.tvec

    # Translation from world frame 1 to world frame 2 in world frame 2
    r_w2w1_w2 = R_c_w2 @ (r_w2c_c - r_w1c_c)

    q_w1_w2 = pycolmap.rotmat_to_qvec(R_w1_w2)

    transform = pycolmap.SimilarityTransform3(scale, q_w1_w2, r_w2w1_w2)

    return transform

def RANSAC_Transformation(results, target_sfm, target, max_it, scale_std, max_distance_error, max_angle_error):
    # Load the estimates for the query poses in the frame of the reference model
    pose_estimates = parse_pose_estimates(results)
    target_model = pycolmap.Reconstruction(target_sfm)

    max_inliers = 0
    best_transform = None
    best_query = ""
    best_scale = 1.0
    best_distance_error = 10000
    best_angle_error = 180

    num_it = 0
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
        pose_in_target1 = target_model.find_image_with_name(f'{target}/' + img_name1)

        # Transform which hopefully aligns the target model with the reference model
        transform1 = calculate_transform(pose_in_target1, pose_est1, scale1)
        target_model_trans_tmp.transform(transform1)


        inliers = 0
        inliers_d = []
        inliers_ang = []
        # 'Inner loop' comparing the transformation found against the other pose estimates
        for img_name2, pose_est2 in pose_estimates.items():

            pose_in_target2 = target_model_trans_tmp.find_image_with_name(f'{target}/' + img_name2)
            
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

    return best_transform
