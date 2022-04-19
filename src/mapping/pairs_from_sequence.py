#!/usr/bin/env python
# coding: utf-8

# Based on https://github.com/cvg/Hierarchical-Localization/pull/175/files
# credit to https://github.com/cduguet


import os
import argparse
import collections.abc as collections
from pathlib import Path
from typing import Dict, List, Union, Optional

from hloc import logger
from hloc.utils.parsers import parse_image_lists, parse_retrieval
from hloc.utils.io import list_h5_names
import pairs_from_retrieval_sequential


def main(
        output: Path,
        image_list: Optional[Union[Path, List[str]]] = None,
        features: Optional[Path] = None,
        window_size: Optional[int] = 5,
        quadratic: bool = False,
        loop_closure: bool = False,
        retrieval_path: Optional[Union[Path, str]] = None,
        N: Optional[int] = 5,
        num_loc: Optional[int] = 5):

    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            print(image_list)
            names_q = parse_image_lists(image_list)
        elif isinstance(image_list, collections.Iterable):
            names_q = list(image_list)
        else:
            raise ValueError(f'Unknown type for image list: {image_list}')
    elif features is not None:
        names_q = list_h5_names(features)
    else:
        raise ValueError('Provide either a list of images or a feature file.')

    pairs = []
    tot = len(names_q)

    for i in range(tot - 1):
        for j in range(i + 1, min(i + window_size, tot)):
            pairs.append((names_q[i], names_q[j]))

            if quadratic:
                q = 2**(j-i)
                if q > window_size and i + q < tot:
                    pairs.append((names_q[i], names_q[i + q]))

    if loop_closure:
        # TODO raise an error if not found!
        # retrieval_path = extract_features.main()

        retrieval_pairs_tmp = output.parent / f'retrieval-pairs-tmp.txt'

        pairs_from_retrieval_sequential.main(
            retrieval_path, retrieval_pairs_tmp, num_matched=num_loc, window_size=window_size)
        
        retrieval = parse_retrieval(retrieval_pairs_tmp)

        for key, val in retrieval.items():
            for match in val:
                pairs.append((key, match))

        os.unlink(retrieval_pairs_tmp)

    logger.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a list of image pairs based on the sequence of images on alphabetic order")
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image_list', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--window_size', type=int, default=5,
                        help="Size of the window of images to match, default: %(default)s")
    parser.add_argument('--quadratic', action="store_true",
                        help="Pair elements with quadratic overlap")
    parser.add_argument('--loop_closure', action="store_true",
                        help="Create a loop sequence (last elements matched with first ones)")
    parser.add_argument('--retrieval_path', type=Path,
                        help="Path to retrieval features, necessary for loop closure")
    parser.add_argument('--N', type=int, default=5,
                        help="Trigger retrieval every N frames, default: %(default)s")
    parser.add_argument('--num_loc', type=int, default=5,
                        help='Number of image pairs for loc, default: %(default)s')
    args = parser.parse_args()
    main(**args.__dict__)
