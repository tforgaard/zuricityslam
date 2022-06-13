import argparse
from pathlib import Path
from filelock import FileLock
import numpy as np
from natsort import natsorted
import networkx as nx
import json

from cityslam import logger
from cityslam.utils.parsers import model_path_2_name, model_name_2_path, find_models, sequential_models
from cityslam.localization import abs_pose_estimation
from cityslam.utils.graph import find_graphs, get_graphs, create_graph_from_model, transform_exists


def main(models, outputs, models_mask=None, only_sequential=False, scores=False, abs_pose_conf={}, overwrite=False, visualize=False):

    # TODO: use pickle to save graphs??

    if isinstance(models_mask,str):
        models_mask = [models_mask]


    outputs = Path(outputs)
    outputs.mkdir(exist_ok=True,parents=True)

    super_graph = find_graphs(models, outputs)
    maps = get_graphs(super_graph)

    if scores:
        scores_file = next(Path(outputs).glob("model_match_scores.json"),None)
        assert scores_file

        min_model_score = 0.3

        with open(scores_file) as f:
            score_dict = json.load(f)

        score_items = score_dict.items()
        tuple_pair = []
        for item in score_items:
            targ = (item[0],)
            sub_dict = item[1]
            tuple_pair += [targ+dict for dict in sub_dict.items()]

        sorted_scores = sorted(tuple_pair, key=lambda item: item[2], reverse=True)

        for (target_name, reference_name, score) in sorted_scores:
            
            reference = model_name_2_path(reference_name)
            target = model_name_2_path(target_name)

            if(score < min_model_score):
                continue
            if target_name == reference_name:
                continue
            if models_mask is not None and reference.parts[0] not in models_mask:
                continue
            if models_mask is not None and target.parts[0] not in models_mask:
                continue
            if only_sequential and not sequential_models(target, reference):
                continue
            if transform_exists(super_graph, target_name, reference_name) and not overwrite:
                continue

            logger.info(f"trying to merge target: {target} and reference {reference}")
            abs_pose_estimation.main(models, outputs, target=target, reference=reference, overwrite=overwrite, visualize=visualize, **abs_pose_conf)

    else:
        merged_models = []
        for map in maps:
            merged_models += map.nodes
        merged_models = [model_name_2_path(m) for m in merged_models]

        all_models = find_models(models, models_mask)

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
                    success = try_merge_model_w_map(models, outputs, model_target, map_ref, abs_pose_conf, overwrite, visualize, models_mask, max_merges=3)

        else:
            for model_ind, model in enumerate(all_models):
                successful_merges = []
                for map_ind, map in enumerate(maps):
                    
                    success = try_merge_model_w_map(models, outputs, model, map, abs_pose_conf, overwrite, visualize, models_mask, max_merges=3)
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


def try_merge_model_w_map(models_dir, output_dir, model, map, abs_pose_conf, overwrite, visualize, models_mask, max_merges=3):
    lock_path = Path(models_dir) / f"{model}.lock"
    lock = FileLock(lock_path)
    with lock:
        merges = 0
        for map_model_name in map.nodes:
            map_model = model_name_2_path(map_model_name)

            if model == map_model:
                continue

            if models_mask is not None and map_model.parts[0] not in models_mask:
                continue

            if transform_exists(map, model_path_2_name(model), map_model_name) and not overwrite:
                continue

            logger.info(f"trying to merge {model} with {map_model}")
            try:
                success = abs_pose_estimation.main(models_dir, output_dir, target=model, reference=map_model, overwrite=overwrite, visualize=visualize, ransac_conf=abs_pose_conf)
                merges += success
            except Exception as e:
                print(e)
            if merges >= max_merges:
                logger.info(f"found max ({max_merges}) amount of merges! Stopping")
                break

    return bool(merges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models-features',
                        help='Path to the models, searched recursively, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/merge',
                        help='Output path, default: %(default)s')
    parser.add_argument('--models_mask', nargs="+", default=None,
                        help='Only include given models: %(default)s')
    parser.add_argument('--only_sequential', action="store_true")
    parser.add_argument('--scores', action="store_true")
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--visualize', action="store_true")
    args = parser.parse_args()

    main(**args.__dict__)
