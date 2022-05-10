# //ZuriCity SLAM Project Group#21 3D-Vision FS2022 ETH///////////////////////
# ============================================================================
## Name        : videointerface.py
# Author      : Tom Heine and Senthuran Kalananthan Group#7 3DV ETH
# Created on  : 29.03.2022
# Description : This sub programm ist tasked to get video based on a search
# query given by a user and to store the results in a file.
# ============================================================================
# ============================================================================

import argparse
from pathlib import Path
import pickle
from matplotlib.font_manager import json_dump

import numpy as np
import what3words as w3w

from .. import logger

# google-API
from apiclient.discovery import build
from apiclient.errors import HttpError

youTubeApiKey = "AIzaSyDZ7EFO3tvl5HKMj4bgQBzvBhbVhDPiCx8"  # Input your youTubeApiKey
w3wApiKey = "AWXNVR24"  # Input your w3wApiKey

# Queries to run through youtube
youtube_queries = ["City Walk", "walk",
                   "Tour", "walking tour", "bike", "driving"]

# JSON capabilities
import json
from datetime import datetime

def main(queries_path, input_type, query, max_results, overwrite=False, verbose=True):

    if not verbose:
        logger.setLevel('ERROR')

    # Get credentials and create an API client
    youtubeAPI = build('youtube', 'v3', developerKey=youTubeApiKey)

    results = {
        'video_id': np.array([], dtype=object),
        'title': np.array([], dtype=object),
        'rank': np.array([], dtype=object),
        'hits': np.array([], dtype=object)
    }

    # get query from user
    # use query to get coordinates
    if input_type == "coordinates":
        location_co = query
    elif input_type == "w3w":
        # get coordinates from w3w
        location_co = w3w_to_CO('///'+query)
    elif input_type == "cityname":
        # get coordinates from google maps
        location_co = cityname_to_CO(query)
    else:
        logger.error("input type not recognized")
        location_co = "47.371667, 8.542222"

    logger.info(f"location coords: <{location_co}>")

    queries = {}
    # check if query is cached
    queries_path = queries_path / "queries.pkl"
    if queries_path.exists() and not overwrite:
        with open(queries_path, 'rb') as file:
            queries = pickle.load(file)

    if query in queries and not overwrite:
        logger.info("Query is already cached, skipping ahead...")
        results = queries[query]

    else:
        # run queries
        for youtube_query in youtube_queries:
            yt_interface(youtubeAPI, youtube_query,
                         results, location_co, max_results)

        # order results found
        order_results(results)

        # store query
        store_query(queries_path, results);

        # cache queries
        # queries[query] = results
        # with open(queries_path, 'wb+') as file:
        #     pickle.dump(queries, file)

    if verbose:
        print_results(results)

    # TODO fix max results !!!! This is temporary hack
    return list(results['video_id'][:max_results])


# //METHODES-GEOPOS-------------------------------------------------------////
def w3w_to_CO(query):
    logger.info("w3w_to_CO from: " + str(query))
    geocoder = w3w.Geocoder(w3wApiKey)
    result = geocoder.convert_to_coordinates(query)
    #print("w3w_to_CO to: " + str(result))
    out = str(result['coordinates']['lat']) + ", " + \
        str(result['coordinates']['lng'])
    logger.info("w3w_to_CO to: " + str(out))
    return out


def cityname_to_CO(cityname):
    logger.info("not ready yet, use Zurich")
    return "47.371667, 8.542222"


# //METHODES-YT-INTERFACE-------------------------------------------------////
def yt_interface(youtubeAPI, query, results, location_co, max_results):
    request = youtubeAPI.search().list(
        part="id,snippet",
        q=query,
        type="video",
        location=location_co,
        locationRadius="25km",
        maxResults=max_results
    )

    response = request.execute()

    for count, video_result in enumerate(response['items']):
        rank = max_results - count
        add_video_to_results(results, video_result, rank)


# //METHODES-DATAMANAGEMENT-----------------------------------------------////
def add_video_to_results(results, video_result, rank):

    if video_result['id']['videoId'] in results['video_id']:
        results['rank'][results['video_id'] ==
                        video_result['id']['videoId']] += rank
        results['hits'][results['video_id'] ==
                        video_result['id']['videoId']] += 1

    else:
        # add new video
        results['video_id'] = np.append(
            results['video_id'], video_result['id']['videoId'])
        results['title'] = np.append(
            results['title'], video_result['snippet']['title'])
        results['rank'] = np.append(results['rank'], rank)
        results['hits'] = np.append(results['hits'], 1)


def order_results(results):

    #results = np.sort(results, order=2)
    permutation = np.argsort(results['rank'])[::-1]
    results['rank'] = results['rank'][permutation]
    results['video_id'] = results['video_id'][permutation]
    results['title'] = results['title'][permutation]
    results['hits'] = results['hits'][permutation]


def print_results(results):

    print("num: videoID     (RNK in Cn) : videoTitle")
    print("_________________________________________")
    for count, video in enumerate(results['video_id']):
        print(str(count).zfill(3) + ": " + str(results['video_id'][count]) + " (" + str(results['rank'][count]).zfill(
            3) + " in " + str(results['hits'][count]).zfill(2) + ")" + " : " + results['title'][count])

# //METHODES-DATASTORAGE--------------------------------------------------////
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def store_query(queries_path, results):
    storagefilename=queries_path.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    resultsforjson = [{'video_id': video_id, 'title': title, 'rank': rank, 'hits': hits} for video_id, title, rank, hits in zip(results['video_id'], results['title'], results['rank'], results['hits'])]
    # for entry in range(results['rank'].size):
    #     for key in results:
    #         json_dump(key + results[key][entry], outfile)
    with open(storagefilename, "w") as outfile:        
        #json.dump(results, outfile, cls=NumpyEncoder)
        json.dump(resultsforjson, outfile)
    pass

# //METHODES-ARGUMENT-HANDLING--------------------------------------------////

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries_path', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/queries',
                        help='folder for video queries')
    parser.add_argument('--input_type', type=str, default='coordinates',  # 'w3w',
                        help='inputtype of the query: %(default)s',
                        choices=['coordinates', 'w3w', 'cityname'])
    parser.add_argument('--query', type=str, default='47.371667, 8.542222',  # 'trailer.sung.believer',
                        help='search query to get video')
    parser.add_argument('--max_results', type=int, default=25,
                        help='Max number of results to fetch')
    parser.add_argument("--verbose",
                        help="increase output verbosity, i.e output result of queries.",
                        action="store_true")
    parser.add_argument("--overwrite",
                        help="Overwrite cached queries",
                        action="store_true")
    args = parser.parse_args()

    video_ids = main(**args.__dict__)
    print(" ".join(video_ids))
