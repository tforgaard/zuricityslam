import argparse
from pathlib import Path
import pickle
from geopy import Nominatim

import numpy as np

from .. import logger

# google-API
from apiclient.discovery import build
from apiclient.errors import HttpError

youTubeApiKey = "AIzaSyDZ7EFO3tvl5HKMj4bgQBzvBhbVhDPiCx8"  # Input your youTubeApiKey

# Queries to run through youtube
youtube_queries = ["City Walk", "walk",
                   "Tour", "walking tour", "bike", "driving"]


def main(queries_path, cityname, num_vids, max_results=100, overwrite=False, verbose=True):

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
    location_co = cityname_to_CO(cityname)
    logger.info(f"location coords: <{location_co}>")

    queries = {}
    # check if query is cached
    queries_path = queries_path / "queries.pkl"
    if queries_path.exists() and not overwrite:
        with open(queries_path, 'rb') as file:
            queries = pickle.load(file)

    queries_path.parent.mkdir(exist_ok=True, parents=True)

    query = cityname
    if query in queries and not overwrite:
        logger.info("Query is already cached, skipping ahead...")
        results = queries[query]

    else:
        # run queries
        for youtube_query in youtube_queries:
            youtube_query = cityname + " " + youtube_query
            yt_interface(youtubeAPI, youtube_query,
                         results, location_co, max_results)

        # order results found
        order_results(results)

        # cache queries
        queries[query] = results
        with open(queries_path, 'wb+') as file:
            pickle.dump(queries, file)

    if verbose:
        print_results(results)

    return list(results['video_id'][:num_vids])

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

def cityname_to_CO(cityname):
    geolocator = Nominatim(user_agent="cityslam")
    location = geolocator.geocode(cityname)
    return str(location.latitude) + ", " + str(location.longitude)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries_path', type=Path,
                        #default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/queries',
                        default='/cluster/home/ksteinsland/zuricityslam/base/kriss/datasets/queries',
                        help='folder for video queries')
    parser.add_argument('--cityname', type=str, default="Zürich",
                        help='input city name: %(default)s')
    parser.add_argument('--num_vids', type=int, default=5,
                        help='Max number of results to fetch')
    parser.add_argument('--max_results', type=int, default=100,
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
