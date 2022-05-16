#!/usr/bin/env python
# coding: utf-8

# Inspired from hloc example SfM_pipeline.py

from pathlib import Path
import argparse

from hloc import extract_features
from . import update_features


def main(images_path, outputs, video_id):

    output_model = outputs / video_id

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']

    output_model.mkdir(exist_ok=True, parents=True, mode=0o777)

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
    image_list = get_images(images_path, subfolder=video_id)
    # image_list = parse_image_list(Path(image_splits) / f"{video_id}_images.txt")
    print(f"num images : {len(image_list)}")
    
    # We extract global descriptors with NetVLAD and find for each image the most similar ones.
    extract_features.main(retrieval_conf, images_path, output_model, image_list=image_list)

    # ## Extract local features
    extract_features.main(feature_conf, images_path, output_model, image_list=image_list)

    # Copy local and global features and from our file to the 'joint' feature files
    # NB! This procedure is blocking for all other processes trying to access the 'joint' feature files
    overwrite=True
    update_features.main(output_model, outputs, overwrite=overwrite)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_path', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/outputs',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--video_id', type=str, default='W25QdyiFnh0',
                        help='video id for subfolder, %(default)s')
    args = parser.parse_args()
    
    main(**args.__dict__)