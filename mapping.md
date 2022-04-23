### Overview

Single video pipeline first extracts image match pairs after sequentiality and/or retrieval. To find retrieval global descriptors must first be extracted using netvlad

Then we extract local features of each image using SuperPoint

Afterwards, we use SuperGlue to match the image pairs 

Lastly we run a reconstruction by first importing the images, features and matches to a colmap type database and then call the incremental mapper of colmap.

### Possible parameters to tweak for better matching

num_loc: number of retrieval matches to find
window_size: number of sequential matches to include
min_num_matches: not completely sure how this works, see colmap


- [ ] Mapping - maybe go back to using only 1fps for faster reconstruction
- [ ] Mapping - create script for rerunning a reconstruction where one tries one uses the largest submodel as a starting point, and for each submodel which is smaller, we fetch MORE matches against the largest model, and then run a continuation reconstruction
- [ ] Mapping - Idea: find where colmap struggles during the mapping and splits the sequence into different models automatically, to extract more images (higher fps) around that area (and or increasing num_loc and window size for that specific area) to increase the success rate of the mapping
- [ ] Mapping - Fetch lates pycolmap from github and build from source!!
- [ ] Mapping - explore parameters for reconstruction!!!

### Walking in ZURICH   Switzerland 🇨🇭- 4K 60fps (UHD).mp4 notes
- seem to be able to map first 200 seconds of the clip just fine as long as we use a high enough framerate, 
    - 2fps, 10 images sequential works
    - 1fps, 10 images sequential does not
    -  need to try reducing the number of  sequential images vs. fps to find optimal number
- need to find out how the download quality of the video affects tracking
    - i.e. 720p is high enough resolution but the bitrate is horrible, might need to download 1080p or 2k/4k and then convert to images and resize, images does not need to be larger than 1600, or 1024
    - 1080p bitrate looks a lot more reasonable, but I think we might need to go higher
- remove first 8 seconds of clip due to watermark
- possible loop closure in 35:26-42:40

