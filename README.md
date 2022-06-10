# ZüriCity-SLAM
## Introduction
ToDo: add this

## Prerequisites
Project is created with
* Python: 3.7

## Set up
Clone this repository:
```
$ git clone --recursive git@gitlab.ethz.ch:z-ricityslam/zuricityslam.git
$ git submodule update --init --recursive
```

Install hloc-fork
```
$ cd hloc_fork
$ pip3 install -r requirements.txt
$ python3 -m pip install -e .
$ cd ..
```

Install required python packages, if they are not already installed:
```
$ pip3 install -r requirements.txt
```
**NB! ffmpeg also needs to be installed for preprocessing**

Then run
```
$ pip3 install -e .
```
to install the cityslam

## Run this project

### Run simple demo of cityslam pipeline
```
$ python3 pipeline.py
```

### To run the intermediate steps of the pipeline

#### Find videos for a given query
```
$ python3 -m cityslam.videointerface.videointerface --query zurich
```

#### Download videos
```
$ python3 -m cityslam.videointerface.downloader --output $VIDEO_PATH --video_ids W25QdyiFnh0 ...
```

#### Preprocess
```
$ python3 -m cityslam.preprocessing.preprocessing --videos $VIDEO_PATH --output $IMAGES_PATH
```

#### Mapping
```
$ python3 -m cityslam.mapping.reconstruction --dataset $SINGLE_VIDEO_PATH
```

#### Merging
```
$ python3 -m cityslam.localization.merge --models $MODELS_PATH --graphs $MERGE_PATH
```

## Euler scripts
See [euler](./euler.md)



## Folder structure

./
├── datasets
│   ├── images
│   ├── image_splits
│   ├── queries
│   ├── transitions
│   ├── transitions_cropped
│   ├── videos
│   └── videos_wv
└── outputs
    ├── merge
    └── models
