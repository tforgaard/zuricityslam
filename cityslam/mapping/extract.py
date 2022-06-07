#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import argparse

from hloc import extract_features
from cityslam.utils.parsers import get_images


def main(images_path, outputs, video_id, overwrite=False):

    output_model = outputs / video_id

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']

    output_model.mkdir(exist_ok=True, parents=True, mode=0o777)

    print("getting images...")
    # Image list is the the relative path to the image from the top most image root folder
    image_list = get_images(images_path, subfolder=video_id)
    # image_list = parse_image_list(Path(image_splits) / f"{video_id}_images.txt")
    print(f"num images : {len(image_list)}")
    
    # We extract global descriptors with NetVLAD and find for each image the most similar ones.
    retrieval_path = extract_features.main(retrieval_conf, images_path, output_model, image_list=image_list, overwrite=overwrite)

    # ## Extract local features
    features_path = extract_features.main(feature_conf, images_path, output_model, image_list=image_list, overwrite=overwrite)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/outputs',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--video_id', type=str, default='W25QdyiFnh0',
                        help='video id for subfolder, %(default)s')
    parser.add_argument('--overwrite', action="store_true")
    args = parser.parse_args()
    
    main(**args.__dict__)