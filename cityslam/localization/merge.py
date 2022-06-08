import argparse
from pathlib import Path

import numpy as np
from natsort import natsorted
import networkx as nx

from cityslam import logger
from cityslam.utils.parsers import model_path_2_name, model_name_2_path, find_models, sequential_models
from cityslam.localization import abs_pose_estimation
from cityslam.graph.utils import find_graphs, get_graphs, create_graph_from_model, transform_exists


def main(models, graphs, models_mask=None, only_sequential=False, abs_pose_conf={}, overwrite=False, visualize=False):

    # TODO: use pickle to save graphs??
    # map_paths = Path(maps_dir).glob("map_*.pkl") 

    graphs = Path(graphs)
    graphs.mkdir(exist_ok=True,parents=True)

    super_graph = find_graphs(models, graphs)
    maps = get_graphs(super_graph)

    merged_models = []
    for map in maps:
        merged_models += map.nodes
    merged_models = [model_name_2_path(m) for m in merged_models]

    all_models = find_models(models, models_mask)
    # unmerged_models = natsorted(list(set(all_models).difference(set(merged_models))))

    if only_sequential:
        for model_target in all_models:
            for model_ref in all_models:
                if model_target == model_ref:
                    continue

                if transform_exists(super_graph, model_path_2_name(model_target), model_path_2_name(model_ref)) and not overwrite:
                    continue

                if not sequential_models(model_target, model_ref):
                    continue

                map_ref = create_graph_from_model(model_ref)
                success = try_merge_model_w_map(models, graphs, model_target, map_ref, abs_pose_conf, overwrite, visualize, max_merges=3)

    else:
        for model_ind, model in enumerate(all_models):
            successful_merges = []
            for map_ind, map in enumerate(maps):
                
                success = try_merge_model_w_map(models, graphs, model, map, abs_pose_conf, overwrite, visualize, max_merges=3)
                if success:
                    # save map?
                    successful_merges.append(map)

            if successful_merges:
                merged_graph = nx.compose_all(successful_merges)
                for suc_map in successful_merges:
                    maps.remove(suc_map)
                maps.append(merged_graph)

            else:
                maps.append(create_graph_from_model(model))


def try_merge_model_w_map(models_dir, output_dir, model, map, abs_pose_conf, overwrite, visualize, max_merges=3):
    merges = 0
    for map_model_name in map.nodes:
        map_model = model_name_2_path(map_model_name)

        logger.info(f"trying to merge {model} with {map_model}")
        success = abs_pose_estimation.main(models_dir, output_dir, target=model, reference=map_model, overwrite=overwrite, visualize=visualize, **abs_pose_conf)
        merges += success
        if merges >= max_merges:
            logger.info(f"found max ({max_merges}) amount of merges! Stopping")
            break

    return bool(merges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models-features',
                        help='Path to the models, searched recursively, default: %(default)s')
    parser.add_argument('--graphs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/merge',
                        help='Output path, default: %(default)s')
    parser.add_argument('--models_mask', nargs="+", default=None,
                        help='Only include given models: %(default)s')
    parser.add_argument('--only_sequential', action="store_true")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()


    # args.only_sequential = True
    # args.models_mask = 'ITntTt4qkWY'

    main(**args.__dict__)
