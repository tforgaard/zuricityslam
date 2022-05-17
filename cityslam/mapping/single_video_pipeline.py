#!/usr/bin/env python
# coding: utf-8

# Inspired from hloc example SfM_pipeline.py

from pathlib import Path
import argparse
from pycolmap import CameraMode


from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
from hloc.utils import viz
from hloc.utils.io import list_h5_names
from hloc.utils.parsers import parse_image_list
from hloc import pairs_from_sequence
from . import update_features


def features_exists(feature_path, images):
    skip_names = set(list_h5_names(feature_path) if feature_path.exists() else ())
    if set(images).issubset(set(skip_names)):
        print('Skipping the extraction.')
        return True
    return False


# Run SfM reconstruction from scratch on a set of images.

# TODO for model merging utilize the loop closure by doing every ten seconds do retrevial to other sequences

confs = {'pairing': ['sequential', 'retrieval', 'sequential+retrieval']}


def main(images_path, image_splits, outputs, video_id, window_size, num_loc, pairing, run_reconstruction, retrieval_interval=5, overwrite=False):

    output_model = outputs / video_id
    sfm_dir = output_model / 'sfm_sp+sg'

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    output_model.mkdir(exist_ok=True, parents=True, mode=0o777)

    # ## Find image pairs either via sequential pairing, image retrieval or eventually both

    def get_images(image_path, subfolder=None):
        globs = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
        image_list = []
        
        if subfolder is not None:
            image_path = image_path / subfolder
        
        for g in globs:
            image_list += list(Path(image_path).glob(g))

        image_list = ["/".join(img.parts[-2:]) for img in image_list]
            
        image_list = sorted(list(image_list))
        return image_list

    print("getting images...")
    # Image list is the the relative path to the image from the top most image root folder
    # image_list = get_images(images_path, subfolder=video_id)
    image_list = parse_image_list(Path(image_splits) / f"{video_id}_images.txt")
    print(f"num images : {len(image_list)}")

    # Check if we find the 'joint' feature files
    joint_retrieval_path = next(outputs.glob("global-feats*.h5"), None)
    joint_feature_path = next(outputs.glob("feats-*.h5"), None)

    if pairing in confs['pairing']:        
        if 'retrieval' in pairing:
            # We extract global descriptors with NetVLAD and find for each image the most similar ones.
            # Check if we find our features in the 'joint' feature file
            if not features_exists(joint_retrieval_path, image_list) or overwrite:
                single_retrieval_path = extract_features.main(
                    retrieval_conf, images_path, output_model, image_list=image_list, overwrite=overwrite)

                # Copy global features and from our file to the 'joint' feature files
                # NB! This procedure is blocking for all other processes trying to access the 'joint' feature files
                joint_retrieval_path = update_features.main(single_retrieval_path, outputs, overwrite)

            single_retrieval_path = next(output_model.glob("global-feats-*.h5"), None)
            if single_retrieval_path is None:
                single_retrieval_path = joint_retrieval_path

        if pairing == 'sequential':
            sfm_pairs = output_model / f'pairs-sequential{window_size}.txt'

            pairs_from_sequence.main(
                sfm_pairs, image_list, features=None, window_size=window_size, quadratic=True)

        elif pairing == 'retrieval':
            sfm_pairs = output_model / f'pairs-retrieval-netvlad{num_loc}.txt'

            pairs_from_retrieval.main(
                single_retrieval_path, sfm_pairs, num_matched=num_loc)

        elif pairing == 'sequential+retrieval':
            sfm_pairs = output_model / f'pairs-sequential{window_size}-retrieval-netvlad{num_loc}.txt'

            pairs_from_sequence.main(
                sfm_pairs, image_list, features=None, window_size=window_size,
                loop_closure=True, quadratic=True, retrieval_path=single_retrieval_path, retrieval_interval=retrieval_interval, num_loc=num_loc)

    else:
        raise ValueError(f'Unknown pairing method')


    # ## Extract and match local features
    # Check if we find our features in the 'joint' feature file
    if not features_exists(joint_feature_path, image_list) or overwrite:
        single_feature_path = extract_features.main(feature_conf, images_path, output_model, image_list=image_list, overwrite=overwrite)

        # Copy local and global features and from our file to the 'joint' feature files
        # NB! This procedure is blocking for all other processes trying to access the 'joint' feature files
        joint_feature_path = update_features.main(single_feature_path, outputs, overwrite)

    # output file for matches
    matches = Path(output_model, f'{joint_feature_path.stem}_{matcher_conf["output"]}_{sfm_pairs.stem}.h5')
    
    # We use our single feature file to allow more parrallel operations
    # if not found default to joint feature file    
    single_feature_path = next(output_model.glob("feats-*.h5"), None)
    if single_feature_path is None:
        single_feature_path = joint_feature_path

    match_path = match_features.main(
        matcher_conf, sfm_pairs, features=single_feature_path, matches=matches, overwrite=overwrite)

    # ## 3D reconstruction
    # Run COLMAP on the features and matches.

    # TODO add camera mode as a param, single works for now, but maybe per folder would be better when we start merging
    model = reconstruction.main(
        sfm_dir, images_path, sfm_pairs, single_feature_path, match_path, image_list=image_list, camera_mode=CameraMode.SINGLE, run=run_reconstruction)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--image_splits', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/image_splits',
                        help='Path to the partioning of the datasets, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/outputs',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--video_id', type=str, default='W25QdyiFnh0',
                        help='video id for subfolder, %(default)s')
    parser.add_argument('--window_size', type=int, default=6,
                        help="Size of the window of images to match sequentially, default: %(default)s")
    parser.add_argument('--num_loc', type=int, default=7,
                        help='Number of image pairs for retrieval, default: %(default)s')
    parser.add_argument('--retrieval_interval', type=int, default=5,
                        help='How often to trigger retrieval: %(default)s')
    parser.add_argument('--pairing', type=str, default='sequential+retrieval',
                        help=f'Pairing method, default: %(default)s', choices=confs['pairing'])
    parser.add_argument('--run_reconstruction', action="store_true",
                        help="If we want to run the pycolmap reconstruction or not")
    parser.add_argument('--overwrite', action="store_true")
    args = parser.parse_args()
    
    # Run mapping
    model = main(**args.__dict__)

    images = args.images_path / args.video_id
    outputs = args.outputs

    if model is not None:
        # We visualize some of the registered images, and color their keypoint by visibility, track length, or triangulated depth.
        
        print("Plotting some examples of sfm points")
        plt_dir = outputs / 'plots'
        plt_dir.mkdir(exist_ok=True, parents=True)

        visualization.visualize_sfm_2d(model, images, color_by='visibility', n=5)
        viz.save_plot(plt_dir / 'visibility.png')
        # plt.show()

        visualization.visualize_sfm_2d(model, images, color_by='track_length', n=5)
        viz.save_plot(plt_dir / 'track_length.png')
        # plt.show()

        visualization.visualize_sfm_2d(model, images, color_by='depth', n=5)
        viz.save_plot(plt_dir / 'depth.png')
        # plt.show()
    
    else:
        print("Model is not created!\n Run hloc or colmap reconstruction!")
