from apiclient.discovery import build #pip install google-api-python-client
from apiclient.errors import HttpError #pip install google-api-python-client
from oauth2client.tools import argparser #pip install oauth2client
from pytube import YouTube

def main():
    # Get credentials and create an API client
    youTubeApiKey="AIzaSyDhCrElzVvJrLSf1R7PAVNJHKYVOpaQDX8" #Input your youTubeApiKey
    youtube=build('youtube','v3',developerKey=youTubeApiKey)

    request = youtube.search().list(
        part="snippet",
        maxResults=25,
        q="Zurich|Zürich"
    )

    response = request.execute()
    video_id = []

    for count, i in enumerate(response['items']):
      video_id.append(i['id']['videoId'])
      yt = YouTube('http://youtube.com/watch?v=' + video_id[count])
      print(yt.title)
      yt.streams.get_highest_resolution().download()
      


if __name__ == "__main__":
    main()