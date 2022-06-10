#!/usr/bin/env python

from pathlib import Path
import argparse
import numpy as np

from hloc import match_features
from hloc import localize_sfm
from hloc.utils import viz_3d
from hloc.utils.parsers import parse_retrieval
import pycolmap

from cityslam.localization.helper_functions import create_query_file, RANSAC_Transformation, default_ransac_conf 
from cityslam.utils.parsers import model_path_2_name, model_name_2_path, get_model_base
from cityslam.utils.features import create_joint_feature_file
from cityslam.localization import model_pairs
from cityslam import logger


def main(models, output_dir, target, reference, ransac_conf = {}, overwrite=False, visualize=False):


    target = Path(target)
    reference = Path(reference)
    
    target_name = model_path_2_name(target)
    reference_name = model_path_2_name(reference)

    outputs = output_dir / target / reference # f'merge_{target_name}__{reference_name}'
    outputs.mkdir(exist_ok=True, parents=True)

    # This is the reference and target model path
    reference_sfm = models / reference
    target_sfm = models / target


    trans_txt = outputs / f'trans__{target_name}__{reference_name}.txt'

    # Retrieval pairs from target to reference
    # Do retrieval from target-queries to reference 
    loc_pairs = next(outputs.glob("pairs*.txt"), None)
    if loc_pairs is None or overwrite:
        loc_pairs, score = model_pairs.main(models, output_dir, target, reference, overwrite=False, visualize=False)
    else:
        logger.info("skipping retrieval, already found")
    
    retrieval = parse_retrieval(loc_pairs)
    query_list = list(retrieval.keys())

    if not query_list:
        logger.info("no pairs")
        logger.info("could not find a transform")
        np.savetxt(trans_txt,[]) 
        return False

    # Results file containing the estimated poses of the queries
    # results = outputs / f'Merge_hloc_superpoint+superglue_netvlad{num_loc}.txt'
    results = outputs / f'Merge_hloc_superpoint+superglue_netvlad.txt'

    # Configurations
    matcher_conf = match_features.confs['superglue']
    
    features_target = next(get_model_base(models, target).glob("feats*.h5"))
    features_ref = next(get_model_base(models, reference).glob("feats*.h5"))

    # Path to joint local features
    # Creating joint feature file necessary for localization
    features_joint = create_joint_feature_file(outputs, models, [target.parts[0], reference.parts[0]], type='features')

    # Create a text file containing query images names and camera parameters
    queries_file = outputs / f'{target_name}_queries_with_intrinsics.txt'
    create_query_file(target_sfm, query_list, queries_file)

    # Output path of matches
    matches = outputs / f'{features_target.stem}_{matcher_conf["output"]}_{loc_pairs.stem}.h5'

    # Do matching for the pairs
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, features=features_target, features_ref=features_ref, matches=matches, overwrite=overwrite)

    # Do localization of queries
    localize_sfm.main(
        reference_sfm,
        queries_file,
        loc_pairs,
        features_joint,
        loc_matches,
        results,
        #ransac_thresh=12,
        covisibility_clustering=False)  
    
    ransac_conf = {**default_ransac_conf, **ransac_conf}

    best_transform = RANSAC_Transformation(results, target_sfm, target, **ransac_conf)    
    
    if best_transform is not None:
        logger.info("found transform, saving it...")
        transform = best_transform.matrix
        np.savetxt(trans_txt, transform[:-1, :], delimiter=',')
        
        reference_model = pycolmap.Reconstruction(reference_sfm)
        target_model_transformed = pycolmap.Reconstruction(target_sfm)
        target_model_transformed.transform(best_transform)

        if visualize:
            fig = viz_3d.init_figure()
            viz_3d.plot_reconstruction(fig, reference_model, color='rgba(255,0,0,0.2)', name="reference")
            viz_3d.plot_reconstruction(fig, target_model_transformed, color='rgba(0,255,0,0.2)', name="target transformed")

            fig.write_html(f'{outputs}/reconstruction__{target_name}__{reference_name}.html')
    
        return True

    else:
        logger.info("could not find a transform")
        np.savetxt(trans_txt,[]) 
        return False
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models',
                        help='Path to the model directory, default: %(default)s')
    parser.add_argument('--output_dir', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/model-pairs',
                        help='Path to the output directory, default: %(default)s')                   
    parser.add_argument('--target', type=str, help='video id for target model')
    parser.add_argument('--reference', type=str, help='video id for reference model')
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--visualize', action="store_true")            
    args = parser.parse_args()
    
    model = main(**args.__dict__)
