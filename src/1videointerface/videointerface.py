##//ZuriCity SLAM Project Group#21 3D-Vision FS2022 ETH///////////////////////
##============================================================================
## Name        : videointerface.py
## Author      : Tom Heine and Senthuran Kalananthan Group#21 3DV ETH
## Created on  : 29.03.2022
## Description : This sub programm ist tasked to get video based on a search 
##               query given by a user and to store the results in a file.
##============================================================================
##============================================================================

##//INCLUDES//////////////////////////////////////////////////////////////////
##//INCLUDE-System-relevant-libs------------------------------------------////
import numpy as np
import argparse
import os
from pathlib import Path
import ffmpeg
import youtube_dl

##//INCLUDE-Api-Client-to-interface-google-API----------------------------////
from apiclient.discovery import build #pip install google-api-python-client
from apiclient.errors import HttpError #pip install google-api-python-client
###TODO: creat installer for pip packagee

##//INCLUDE-Api-Client-to-interface-w3w-API-------------------------------////
import what3words as w3w #pip install what3words

##//INCLUDE-pytube-to-downlode-from-youtube-------------------------------////
from pytube import YouTube #pip install pytube

##//INCLUDE-Aditionals----------------------------------------------------////
from oauth2client.tools import argparser #pip install oauth2client

##//DEFINES///////////////////////////////////////////////////////////////////

##//METHODES//////////////////////////////////////////////////////////////////

##//METHODE-MAIN----------------------------------------------------------////
def main(args):
    ##//VARIABLES/////////////////////////////////////////////////////////////
    #TODO: if someone knows a better way to place global variables please change it
    ##//VARIABLES-APIs----------------------------------------------------////
    # Get credentials and create an API client
    global youTubeApiKey
    youTubeApiKey="AIzaSyBh0zDXZq44mxmMq2p7eUhZnad_ElXgA6A" #Input your youTubeApiKey
    global youtubeAPI
    youtubeAPI=build('youtube','v3',developerKey=youTubeApiKey)

    global w3wApiKey
    w3wApiKey="***REMOVED***" #Input your w3wApiKey

    global thisMaxResults 
    thisMaxResults = 25
    global thisLocationCO
    thisLocationCO = "47.371667, 8.542222"

    ##//VARIABLES-QUERY-PROCESSING----------------------------------------////
    #global results
    #[0] Video ID
    #[1] Video Title
    #[2] Rank    
    #results = np.array([[]],dtype=object)
    ###TODO: this would be a nice solution but python hates me :/
    global results_videoID
    results_videoID = np.array([],dtype=object)
    global results_videoTitle
    results_videoTitle = np.array([],dtype=object)
    global results_rank
    results_rank = np.array([],dtype=object)
    global result_hits 
    result_hits = np.array([],dtype=object)

    #results=np.append(results, [0,"initalising",0])
    results_videoID = np.append(results_videoID, "1234567890A")
    results_videoTitle = np.append(results_videoTitle, "initalising")
    results_rank = np.append(results_rank, 0)
    result_hits = np.append(result_hits, 0)

    ##//Main-Code---------------------------------------------------------////

    #get query from user
    inputtype = args.input_type
    input = args.query

    #use query to get coordinates
    if inputtype == "coordinates":
        thisLocationCO = input
    elif inputtype == "w3w":
        #get coordinates from w3w
        thisLocationCO = w3w_to_CO('///'+input)
    elif inputtype == "cityname":
        #get coordinates from google maps
        thisLocationCO = cityname_to_CO(input)
    else:
        print("ERROR: inputtype not recognized")
        thisLocationCO = "47.371667, 8.542222"

    print("thisLocationCO: <" + str(thisLocationCO) + ">")

    #run queries
    yt_interface("City Walk")
    yt_interface("walk")
    yt_interface("Tour")
    yt_interface("walking tour")
    yt_interface("bike")
    yt_interface("driving")

    #order results found
    order_results()
    #print_results()

    #store results to files
    #download_videos(args.base_dir, args.download_fol) 
    preprocessing(args.base_dir, args.download_fol) 
    #TODO: test implementation and add filter for only multiple hits / good ranking
      
##//METHODES-GEOPOS-------------------------------------------------------////
def w3w_to_CO(input):
    print("w3w_to_CO from: " + str(input))
    geocoder = w3w.Geocoder(w3wApiKey)
    result = geocoder.convert_to_coordinates(input)
    #print("w3w_to_CO to: " + str(result))
    out = str(result['coordinates']['lat']) + ", " + str(result['coordinates']['lng'])
    print("w3w_to_CO to: " + str(out))
    return out


# {'country': 'CH', 
# 'square': {
#         'southwest': {'lng': 8.541667, 'lat': 47.376887}, 
#         'northeast': {'lng': 8.541707, 'lat': 47.376914}
#         }, 
# 'nearestPlace': 'Zürich (Kreis 1) / Lindenhof, Zurich', 
# 'coordinates': {'lng': 8.541687, 'lat': 47.3769}, 
# 'words': 'trailer.sung.believer', 
# 'language': 'en',
# 'map': 'https://w3w.co/trailer.sung.believer'
# }

def cityname_to_CO(cityname):
    print("not ready yet, use Zurich")
    return "47.371667, 8.542222"


##//METHODES-YT-INTERFACE-------------------------------------------------////
def yt_interface(query):
    request = youtubeAPI.search().list(
        part="id,snippet",
        q=query,
        type="video",
        location=thisLocationCO,
        locationRadius="25km",
        maxResults=thisMaxResults
    )

    response = request.execute()
    video_id = []

    for count, video in enumerate(response['items']):
        rank = thisMaxResults - count
        add_video_to_result(video, rank)    
    pass

##//METHODES-DATAMANAGEMENT-----------------------------------------------////
def add_video_to_result(result, rank):
    global results_videoID
    global results_videoTitle
    global results_rank
    global result_hits

    if result['id']['videoId'] in results_videoID:
        results_rank[results_videoID == result['id']['videoId']] += rank
        result_hits[results_videoID == result['id']['videoId']] += 1
    else:
        #add new video
        #results = np.append(results, [result['id']['videoId'], result['snippet']['title'] , rank])
        results_videoID = np.append(results_videoID, result['id']['videoId'])
        results_videoTitle = np.append(results_videoTitle, result['snippet']['title'])
        results_rank = np.append(results_rank, rank)
        result_hits = np.append(result_hits, 1)
    pass

def order_results():
    global results_videoID
    global results_videoTitle
    global results_rank
    global result_hits

    #results = np.sort(results, order=2)
    permutation = np.argsort(results_rank)[::-1]
    results_rank = results_rank[permutation]
    results_videoID = results_videoID[permutation]
    results_videoTitle = results_videoTitle[permutation]
    result_hits = result_hits[permutation]
    pass

def print_results():
    global results_videoID
    global results_videoTitle
    global results_rank
    global result_hits

    #for count, video in enumerate(results[0]):
        #print(str(count) + ": " + results[1][count] + " : " + results[1][count] + " (" + results[2][count] + ")")
    print("num: videoID     (RNK in Cn) : videoTitle")
    print("_________________________________________")
    for count, video in enumerate(results_videoID):
        print(str(count).zfill(3) + ": " + str(results_videoID[count]) + " (" + str(results_rank[count]).zfill(3) + " in " + str(result_hits[count]).zfill(2) + ")" + " : " + results_videoTitle[count])
    pass

##//METHODES-YT-DOWNLOADE-------------------------------------------------////
def download_videos(base_dir, folder):
    
    path = os.path.join(base_dir, folder)
    Path(path).mkdir(parents=True, exist_ok=True)
    for video in results_videoID:
        yt = YouTube('http://youtube.com/watch?v=' + video)
        yt.streams.get_highest_resolution().download(path)
    
    """
    ydl_opts = {
    'format': 'bestvideo/best',
    'videoformat':'mp4',
    'outtmpl': path + '/%(id)s',
    'noplaylist' : True,        
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        for video in results_videoID:
            ydl.download(['https://www.youtube.com/watch?v='+ video])
            break
    """

##//METHODES-PREPROESSING-------------------------------------------------////
def frame_capture(video_path, image_folder):
    try:
        ffmpeg.input(video_path) \
            .filter('fps', fps='2') \
            .output(image_folder+'/img-%d.jpg', start_number=0) \
            .overwrite_output() \
            .run(quiet=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e

def preprocessing(base_dir, folder):
    path = os.path.join(base_dir, folder)

    for count, file in enumerate(os.listdir(path)):
        if file.endswith(".mp4"):
            video_path = os.path.join(path, file) # video_path
            image_folder = "video" + str(count) # folder name for the images 
            image_path = os.path.join(base_dir, "datasets", image_folder)
            Path(image_path).mkdir(parents=True, exist_ok=True)
            frame_capture(video_path, image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=Path,
                        default='/Users/Senthuran/Desktop/Master_ETH/3D_Vision/zuricityslam',
                        help='base directory for datasets and outputs, default: %(default)s')
    parser.add_argument('--download_fol', type=Path,
                        default='videos',
                        help='folder name for downloading videos')                    
    parser.add_argument('--input_type', type=str, default='w3w',
                        help='inputtype of the query: coordinates, w3w, cityname')
    parser.add_argument('--query', type=str, default='trailer.sung.believer',
                        help='search query to get video')
    args = parser.parse_args()

    main(args)
