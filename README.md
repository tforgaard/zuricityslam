# ZüriCity-SLAM
## Introduction
ToDo: add this

## Prerequisites
Project is created with 
* Python: 3.9.8

## Set up
Clone this repository:
```
$ git clone git@gitlab.ethz.ch:z-ricityslam/zuricityslam.git
```

Install required python packages, if they are not already installed:
```
$ pip3 install -r requirements.txt    
```
There has been a bug in the pytube library since 17.4.22. In order to fix the bug, please follow the instructions from eroc1234: https://stackoverflow.com/questions/68945080/pytube-exceptions-regexmatcherror-get-throttling-function-name-could-not-find. 

TODO: Pytube works, but look for alternative if there is no new release

## Run this project
TODO:
```
$ python3 src/1videointerface/videointerface.py --base_dir $path --input_type coordinates --query 47.371667,8.542222
```

## ToDo-List:

- [ ] video interface - Place search (TH)
- [x] video interface - Download method clean up (SH)
- [x] video interface - Installer for Python libraries (SH)
- [x] video interface - Interface Method (SH)
- [ ] Pre-Processing Frame Splitting (TH)
- [ ] Pre-Processing opt. Transition Splitting (tA)
- [x] Mapping - Add sequential pairing (KS\\/TF)
- [x] Mapping - Make callable main function (KS\\/TF)
- [x] Mapping - Merge sequential pairing and retrieval (global) pairing (KS\\/TF)
- [ ] Mapping - Make optimized sequential and retrieval (global) pairing (KS\\/TF)
- [ ] Mapping - Sucessfully run mapping on a complete zurich city walk sequence (KS\\/TF)
- [ ] pipeline - global call method (TH,KS\\/TF)

- [x] adding Tasks to ToDo List(KS/\\TF)
- [ ] add necesseary requirements for this project (tA)
- [ ] make it an installable project (tA)
- [ ] try to follow [PEP8 style guidelines](https://peps.python.org/pep-0008/), install i.e. autopep8 (tA)
- [ ] MTP - presentation slides ready (A)

## Meetings:
[Supervisor Meeting 20220315](docu/meeting20220315.md)  
[Team Meeting 20220315](docu/teammeeting20220315.md)  
[Supervisor Meeting 20220412](docu/meeting20220412.md)

## Abbreviations:
KS Kristoffer Steinsland
SK Senthuran Kalananthan
TF Theodor Forgaard
TH Tom Heine

tA to Assign
A  all
