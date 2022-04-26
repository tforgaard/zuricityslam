#!/usr/bin/env python
# coding: utf-8

# Inspired from hloc example SfM_pipeline.py

from pathlib import Path
import argparse
from pycolmap import CameraMode

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
from hloc.utils import viz
from hloc import pairs_from_sequence

# Run SfM reconstruction from scratch on a set of images.

# TODO for model merging utilize the loop closure by doing every ten seconds do retrevial to other sequences

confs = {'pairing': ['sequential', 'retrieval', 'sequential+retrieval']}


def main(images, outputs, window_size, num_loc, pairing, run_reconstruction):

    sfm_dir = outputs / 'sfm_sp+sg'

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    outputs.mkdir(exist_ok=True, parents=True, mode=0o777)

    # ## Find image pairs either via sequential pairing, image retrieval or eventually both

    def get_image_names(image_path):
        image_list = list(Path(image_path).glob('*.jpg'))
        image_list = [str(il.name) for il in image_list]
        image_list = sorted(list(set(image_list)))
        return image_list
        
    if pairing in confs['pairing']:
        if pairing == 'sequential':
            image_list = get_image_names(images)

            sfm_pairs = outputs / f'pairs-sequential{window_size}.txt'

            pairs_from_sequence.main(
                sfm_pairs, image_list, features=None, window_size=window_size, quadratic=True)
        elif pairing == 'retrieval':
            # We extract global descriptors with NetVLAD and find for each image the most similar ones.
            retrieval_path = extract_features.main(
                retrieval_conf, images, outputs)

            sfm_pairs = outputs / f'pairs-retrieval-netvlad{num_loc}.txt'

            pairs_from_retrieval.main(
                retrieval_path, sfm_pairs, num_matched=num_loc)

        elif pairing == 'sequential+retrieval':
            # We extract global descriptors with NetVLAD and find for each image the most similar ones.
            retrieval_path = extract_features.main(
                retrieval_conf, images, outputs)

            image_list = get_image_names(images)

            sfm_pairs = outputs / f'pairs-sequential{window_size}-retrieval-netvlad{num_loc}.txt'

            pairs_from_sequence.main(
                sfm_pairs, image_list, features=None, window_size=window_size,
                loop_closure=True, quadratic=True, retrieval_path=retrieval_path, retrieval_interval=5, num_loc=num_loc)

    else:
        raise ValueError(f'Unknown pairing method')

    # ## Extract and match local features

    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    # ## 3D reconstruction
    # Run COLMAP on the features and matches.

    # TODO add camera mode as a param, single works for now, but maybe per folder would be better when we start merging
    model = reconstruction.main(
        sfm_dir, images, sfm_pairs, feature_path, match_path, camera_mode=CameraMode.SINGLE, run=run_reconstruction)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images/W25QdyiFnh0',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/outputs/W25QdyiFnh0',
                        help='Path to the output directory, default: %(default)s')
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
    args = parser.parse_args()
    
    # Run mapping
    model, outputs, images = main(**args.__dict__)

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
