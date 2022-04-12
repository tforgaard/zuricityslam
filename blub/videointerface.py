##//ZuriCity SLAM Project Group#21 3D-Vision FS2022 ETH///////////////////////
##============================================================================
## Name        : videointerface.py
## Author      : Tom Heine and Senthuran //TODO: add name Group#21 3DV ETH
## Created on  : 29.03.2022
## Description : This sub programm ist tasked to get video based on a search 
##               query given by a user and to store the results in a file.
##============================================================================
##============================================================================

##//INCLUDES//////////////////////////////////////////////////////////////////
##//INCLUDE-System-relevant-libs------------------------------------------////
import numpy as np

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
def main():
    ##//VARIABLES/////////////////////////////////////////////////////////////
    #TODO: if someone knows a better way to place global variables please change it
    ##//VARIABLES-APIs----------------------------------------------------////
    # Get credentials and create an API client
    global youTubeApiKey
    youTubeApiKey="AIzaSyDhCrElzVvJrLSf1R7PAVNJHKYVOpaQDX8" #Input your youTubeApiKey
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
    inputtype = input("Select inputtype: ")
    #inputtype = "w3w"
    #input = "///trailer.sung.believer"
    input_w =  input("Enter " + inputtype + ": ")
    #use query to get coordinates
    if inputtype == "coordinates":
        thisLocationCO = input_w
    elif inputtype == "w3w":
        #get coordinates from w3w
        thisLocationCO = w3w_to_CO(input_w)
    elif inputtype == "cityname":
        #get coordinates from google maps
        thisLocationCO = cityname_to_CO(input_w)
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
    print_results()

    #store results to files
    #download_videos() 
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
    #if np.any(results[0] == result['id']['videoId']):
        #video allready known improve rank
        #results[2][results[0] == result] += rank
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

def download_videos():
    for count, video in enumerate(results_videoID):
        yt = YouTube('http://youtube.com/watch?v=' + video)
        yt.streams.get_highest_resolution().download()
        break
    pass
    """
    for count, video in enumerate(results[0]):
        yt = YouTube('http://youtube.com/watch?v=' + video[0][count])
        yt.streams.get_highest_resolution().download()
        break
    pass
    """

if __name__ == "__main__":
    main()