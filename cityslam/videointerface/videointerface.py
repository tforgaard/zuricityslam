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

def main(queries_path, blacklist_path, input_type, query, max_results, overwrite=False, verbose=True):

    if not verbose:
        logger.setLevel('ERROR')

    # Get credentials and create an API client
    youtubeAPI = build('youtube', 'v3', developerKey=youTubeApiKey)
    #TODO bug that YT API USE is done when there is nothing to do

    results = {
        'video_id': np.array([], dtype=object),
        'title': np.array([], dtype=object),
        'rank': np.array([], dtype=object),
        'hits': np.array([], dtype=object),
        'location_co': np.array([], dtype=object)
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
    #queries_path = queries_path / "queries.pkl"
    print(queries_path)
    if queries_path.exists() and not overwrite:
        with open(queries_path, 'r') as openfile:
            json_object = json.load(openfile)

        #read queries from file
        if location_co in json_object['location_co'] and not overwrite:
            logger.info("Query is already cached, skipping ahead...")            
            load_results_form_JSON_object(json_object, location_co, results)
    else:
        logger.info("Query is not done or overwritten, starting new search...")
        # run queries
        for youtube_query in youtube_queries:
            yt_interface(youtubeAPI, youtube_query,
                        results, location_co, max_results)

        # order results found
        order_results(results)

    # remove excluded videos
    removeblacklist(results, blacklist_path)

    # store query
    store_query(queries_path, results)

    if verbose:
        print_results(results)
        pass

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
        add_video_to_results(results, video_result, rank, location_co)


# //METHODES-DATAMANAGEMENT-----------------------------------------------////
def add_video_to_results(results, video_result, rank, location_co):

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
        results['location_co'] = np.append(results['location_co'], location_co)


def order_results(results):

    #results = np.sort(results, order=2)
    permutation = np.argsort(results['rank'])[::-1]
    results['rank'] = results['rank'][permutation]
    results['video_id'] = results['video_id'][permutation]
    results['title'] = results['title'][permutation]
    results['hits'] = results['hits'][permutation]


def print_results(results):

    print("num: videoID     (RNK in Cn) : videoTitle ; query")
    print("_________________________________________________")
    for count, video in enumerate(results['video_id']):
        print(str(count).zfill(3) + ": " + str(results['video_id'][count]) + " (" + str(results['rank'][count]).zfill(
            3) + " in " + str(results['hits'][count]).zfill(2) + ")" + " : " + results['title'][count] + " ; " + results['location_co'][count])

# //METHODES-DATASTORAGE--------------------------------------------------////
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def store_query(queries_path, results, humanformat=False):
    storagefilename=queries_path
    if humanformat:
        resultsforjson = [{'video_id': video_id, 'title': title, 'rank': rank, 'hits': hits, 'location_co': location_co} for video_id, title, rank, hits, location_co in zip(results['video_id'], results['title'], results['rank'], results['hits'], results['location_co'])]
    # for entry in range(results['rank'].size):
    #     for key in results:
    #         json_dump(key + results[key][entry], outfile)
    with open(storagefilename, "w+") as outfile:        
        if humanformat:
            json.dump(resultsforjson, outfile)
        else:
            json.dump(results, outfile, cls=NumpyEncoder)
    pass


def load_results_form_JSON_object(json_object, location_co ,results):
    allstoredresults = {
                'video_id': np.array([], dtype=object),
                'title': np.array([], dtype=object),
                'rank': np.array([], dtype=object),
                'hits': np.array([], dtype=object),
                'location_co': np.array([], dtype=object)
    }
    allstoredresults['video_id']=json_object['video_id']
    allstoredresults['title']=json_object['title']
    allstoredresults['rank']=json_object['rank']
    allstoredresults['hits']=json_object['hits']
    allstoredresults['location_co']=json_object['location_co']
    for count, temp_loc in enumerate(allstoredresults['location_co']):
        if temp_loc != (location_co):
            allstoredresults['video_id'].delete(count)
            allstoredresults['title'].delete(count)
            allstoredresults['rank'].delete(count)
            allstoredresults['hits'].delete(count)
            allstoredresults['location_co'].delete(count)
    results = allstoredresults

    # cache queries
    # queries[query] = results
    # with open(queries_path, 'wb+') as file:
    #     pickle.dump(queries, file)

def removeblacklist(results, blacklist_path):
    #if results['video_id'].size > 0:
        with open(blacklist_path, 'r') as openfile:
            blacklist = json.load(openfile)
        for video_id in blacklist['video_id']:
            if video_id in results['video_id']:
                idx = np.where(results['video_id'] == video_id)   
                #print(video_id, "is blacklisted ", idx)
                results['video_id'] = np.delete(results['video_id'],(idx))
                results['title'] = np.delete(results['title'],(idx))
                results['rank'] = np.delete(results['rank'],(idx))
                results['hits'] = np.delete(results['hits'],(idx))
                results['location_co'] = np.delete(results['location_co'],(idx))


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
