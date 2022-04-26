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
import os
from pathlib import Path
import pickle

import numpy as np
import yt_dlp
import ffmpeg
import what3words as w3w
from oauth2client.tools import argparser

# google-API
from apiclient.discovery import build
from apiclient.errors import HttpError

youTubeApiKey = "AIzaSyDZ7EFO3tvl5HKMj4bgQBzvBhbVhDPiCx8"  # Input your youTubeApiKey
w3wApiKey = "AWXNVR24"  # Input your w3wApiKey

# Queries to run through youtube
youtube_queries = ["City Walk", "walk",
                   "Tour", "walking tour", "bike", "driving"]


def main(videos_path, images_path, input_type, query, max_results, num_vids, fps=2, format="bv", overwrite=False, verbose=False, start='00:00:00', duration='00:00:00'):

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
        print("ERROR: input type not recognized")
        location_co = "47.371667, 8.542222"

    print(f"location coords: <{location_co}>")

    queries = {}
    # check if query is cached
    queries_path = Path("queries.pkl")
    if queries_path.exists():
        with open(queries_path, 'rb') as file:
            queries = pickle.load(file)

    if query in queries and not overwrite:
        print("Query is already cached, skipping ahead...")
        results = queries[query]

    else:
        # run queries
        for youtube_query in youtube_queries:
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


    # store results to files
    download_videos(videos_path, results, num_vids, format)
    image_dirs = preprocessing(videos_path, images_path, num_vids, overwrite, fps, start, duration)
    # TODO: test implementation and add filter for only multiple hits / good ranking

    return image_dirs


# //METHODES-GEOPOS-------------------------------------------------------////
def w3w_to_CO(query):
    print("w3w_to_CO from: " + str(query))
    geocoder = w3w.Geocoder(w3wApiKey)
    result = geocoder.convert_to_coordinates(query)
    #print("w3w_to_CO to: " + str(result))
    out = str(result['coordinates']['lat']) + ", " + \
        str(result['coordinates']['lng'])
    print("w3w_to_CO to: " + str(out))
    return out


def cityname_to_CO(cityname):
    print("not ready yet, use Zurich")
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


# //METHODES-YT-DOWNLOADE-------------------------------------------------////
def download_videos(video_path, results, num_vids, format):
    video_path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        'format': format,
        'paths': {'home': f'{video_path}'},  # home is download directory...
        'output': {'home': '%(id)s'}  #
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for count, video in enumerate(results['video_id']):
            if count >= num_vids:
                break
            ydl.download(['https://www.youtube.com/watch?v=' + video])


# //METHODES-PREPROESSING-------------------------------------------------////
def frame_capture(video_path, images_path, prefix="", fps=2, start='00:00:00', duration='00:00:00'):
    if prefix:
        prefix = prefix + "_"
    
    input_opts = {'ss': start}
    if duration != '00:00:00':
        input_opts = {'t': duration}
    
    try:
        ffmpeg.input(video_path, **input_opts) \
            .filter('fps', fps=f'{fps}') \
            .output(str(images_path / f'{prefix}img_fps{fps}_%05d.jpg'), start_number=0) \
            .overwrite_output() \
            .run(quiet=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e


def preprocessing(videos_path, output_path, num_vids, overwrite, fps, start, duration):

    image_dirs = []
    for count, file in enumerate(os.listdir(videos_path)):
        if file.endswith((".mp4", "webm")):
            if count >= num_vids:
                break
            print(f"extracting frames from video: {file}")
            print(f"using fps {fps}")
            video_id = file[file.find("[")+1:file.find("]")]
            video_path = videos_path / file  # video_path
            images_path = output_path / f"{video_id}" 
            image_dirs.append(images_path)
            images_path.mkdir(parents=True, exist_ok=True)
            if list(images_path.glob(f"*_fps{fps}_*.jpg")) and not overwrite:
                print(f"frames already extracted for video: {file}")
                continue
            frame_capture(video_path, images_path, prefix=video_id, fps=fps, start=start, duration=duration)

    return image_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_path', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/videos',
                        help='folder for downloading videos')
    parser.add_argument('--images_path', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images',
                        help='folder for preprocessed images')
    parser.add_argument('--input_type', type=str, default='coordinates',  # 'w3w',
                        help='inputtype of the query: %(default)s',
                        choices=['coordinates', 'w3w', 'cityname'])
    parser.add_argument('--query', type=str, default='47.371667, 8.542222',  # 'trailer.sung.believer',
                        help='search query to get video')
    parser.add_argument('--max_results', type=int, default=25,
                        help='Max number of results to fetch')
    parser.add_argument('--num_vids', type=int, default=1,
                        help='Number of videos to download')
    parser.add_argument('--format', type=str, default='wv',
                        help='Download format to fetch, default: %(default)s (select best video)',
                        choices=['bv', 'wv'])  # and more!
    parser.add_argument('--fps', type=int, default='2',
                        help='fps to use in image splitting')
    parser.add_argument("--verbose",
                        help="increase output verbosity, i.e output result of queries.",
                        action="store_true")
    parser.add_argument("--overwrite",
                        help="Overwrite cached queries",
                        action="store_true")
    args = parser.parse_args()

    main(**args.__dict__)
