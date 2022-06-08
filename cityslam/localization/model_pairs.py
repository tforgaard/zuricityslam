import argparse
from pathlib import Path
import numpy as np

from hloc import pairs_from_retrieval_resampling
from hloc.utils.parsers import parse_retrieval

from cityslam import logger
from cityslam.utils.parsers import get_images_from_recon, get_model_base


default_model_pair_conf = {
    'num_loc' : 10,
    'retrieval_interval' : 15,
    'min_retrieval_score' : 0.15,
    'resample_runs' : 3,
}

def main(models, output, target, reference , conf={}, overwrite=False, visualize=False, **kwargs):

    target = Path(target)
    reference = Path(reference)

    outputs = output / target / reference # f'merge_{target_name}__{reference_name}'
    outputs.mkdir(exist_ok=True, parents=True)

    # This is the reference and target model path
    reference_sfm = models / reference
    target_sfm = models / target

    conf = {**default_model_pair_conf, **conf, **kwargs}

    # Retrieval pairs from target to reference
    loc_pairs = outputs / f'pairs-merge-{conf["num_loc"]}.txt' # top-k retrieved by NetVLAD


    # Paths to local decriptors and features
    descriptor_ref = next(get_model_base(models, reference).glob("global-feats*.h5"))
    descriptor_target = next(get_model_base(models, target).glob("global-feats*.h5"))


    # Get images from target model
    img_names_ref = get_images_from_recon(reference_sfm)
    img_names_target = get_images_from_recon(target_sfm)
    queries = img_names_target


    logger.info(f"finding pairs for model {target} and {reference}")

    
    # Do retrieval from target-queries to reference 
    if not loc_pairs.exists() or overwrite:

        pairs = []
        # Check to see if models are sequential partitions
        if get_model_base(models, target) == get_model_base(models, reference):
            pairs = check_for_common_images(
                img_names_target, img_names_ref, target, reference)

        # Mask out already matched pairs
        match_mask = np.zeros(
            (len(queries), len(img_names_ref)), dtype=bool)
        for (p1, p2) in pairs:
            if p1 in queries:
                match_mask[queries.index(
                    p1), img_names_ref.index(p2)] = True

        # Find retrieval pairs
        _, score = pairs_from_retrieval_resampling.main(descriptor_target, loc_pairs,
                                                        num_matched=conf['num_loc'], query_list=queries,
                                                        db_model=reference_sfm, db_descriptors=descriptor_ref,
                                                        min_score=conf['min_retrieval_score'], match_mask=match_mask,
                                                        query_interval=conf['retrieval_interval'], resample_runs=conf['resample_runs'],
                                                        visualize=visualize)

        # Add common image pairs
        retrieval = parse_retrieval(loc_pairs)

        for key, val in retrieval.items():
            for match in val:
                if (key, match) not in pairs:
                    pairs.append((key, match))

        with open(loc_pairs, 'w') as f:
            f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

        return loc_pairs, score

    else:
        logger.info("skipping retrieval, already found")
        retrieval = parse_retrieval(loc_pairs)
        query_list = list(retrieval.keys())
        
        return loc_pairs, None

def check_for_common_images(img_names_target, img_names_ref, target, reference):

    pairs = []

    seq_n_target = int(target.parts[1].split("part")[-1])
    seq_n_ref = int(reference.parts[1].split("part")[-1])

    if seq_n_target + 1 == seq_n_ref or seq_n_target - 1 == seq_n_ref:
        # look for same images in the two reconstructions..
        img_stems_target = set([img_name_target.split("/")[-1]
                               for img_name_target in img_names_target])
        img_stems_ref = set([img_name_ref.split("/")[-1]
                            for img_name_ref in img_names_ref])

        img_stems_common = img_stems_target.intersection(img_stems_ref)

        for img_stem_common in img_stems_common:

            pairs.append(("/".join([target.parts[0], img_stem_common]),
                          "/".join([reference.parts[0], img_stem_common])))

        logger.info(f"found {len(pairs)} pairs from common images")
    return pairs



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--models_dir', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models',
#                         help='Path to the models, searched recursively, default: %(default)s')
#     parser.add_argument('--outputs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/model-pairs',
#                         help='Output path, default: %(default)s')
#     parser.add_argument('--num_loc', type=int, default=10,
#                         help='Number of image pairs for retrieval, default: %(default)s')
#     parser.add_argument('--retrieval_interval', type=int, default=15,
#                         help='How often to trigger retrieval: %(default)s')
#     parser.add_argument('--resample_runs', type=int, default=3,
#                         help='How often to trigger retrieval: %(default)s')
#     parser.add_argument('--min_score', type=float, default=0.1,
#                         help='Minimum score for retrieval: %(default)s')
#     parser.add_argument('--models_mask', nargs="+", default=None,
#                         help='Only include given models: %(default)s')
#     parser.add_argument('--overwrite', action="store_true")
#     parser.add_argument('--visualize', action="store_true")
#     args = parser.parse_args()

#     main(**args.__dict__)
