#!/usr/bin/env python
# coding: utf-8

# Inspired from hloc example SfM_pipeline.py

from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from hloc import extract_features, match_features, reconstruction, visualization, pairs_from_retrieval
from hloc.utils import viz

# Run SfM reconstruction from scratch on a set of images.

# TODO for model merging utilize the loop closure by doing every ten seconds do retrevial to other sequences


def main(base_dir, dataset, outputs, num_loc):

    # define paths
    images = base_dir / dataset / 'images'
    outputs = base_dir / outputs

    sfm_pairs = outputs / f'pairs-netvlad{num_loc}.txt'
    sfm_dir = outputs / 'sfm_superpoint+superglue'

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    outputs.mkdir(exist_ok=True, parents=True)
    # ref_sfm = outputs / 'sfm_superpoint+superglue'
    # ref_sfm_scaled = outputs / 'sfm_sift_scaled'
    # query_list = outputs / 'query_list_with_intrinsics.txt'
    # sfm_pairs = outputs / f'pairs-db-covis{num_covis}.txt'
    # loc_pairs = outputs / f'pairs-query-netvlad{num_loc}.txt'

    # ## Find image pairs via image retrieval
    # We extract global descriptors with NetVLAD and find for each image the most similar ones. For smaller dataset we can instead use exhaustive matching via `hloc/pairs_from_exhaustive.py`, which would find $\frac{n(n-1)}{2}$ images pairs.

    # TODO increase number of num mathced
    retrieval_path = extract_features.main(retrieval_conf, images, outputs)
    # TODO make a new pairs_from_retrieval that takes sequenciality (±10 images) and also does retrieval but NOT on the pairs already found from sequenciality
    pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=num_loc)

    # ## Extract and match local features

    feature_path = extract_features.main(feature_conf, images, outputs)
    match_path = match_features.main(
        matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    # ## 3D reconstruction
    # Run COLMAP on the features and matches.

    # TODO add camera mode not equal to AUTO!
    model = reconstruction.main(
        sfm_dir, images, sfm_pairs, feature_path, match_path)

    return model, outputs, images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07',
                        help='base directory for datasets and outputs, default: %(default)s')
    parser.add_argument('--dataset', type=Path, default='datasets/long_walk_zurich',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='outputs/long_walk_zurich',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=20,
                        help='Number of image pairs for loc, default: %(default)s')
    args = parser.parse_args()
    model, outputs, images = main(**args.__dict__)

    # ## Visualization
    # We visualize some of the registered images, and color their keypoint by visibility, track length, or triangulated depth.

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
