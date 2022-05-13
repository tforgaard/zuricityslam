#!/usr/bin/env python

#from transnetv2 import TransNetV2
import argparse
from pathlib import Path
import numpy as np
from transnetv2 import TransNetV2

def main(video_file_path, images_path, model_path, output_cuts_path, output_folders_path):
    """
    Finds transitions in videos, divides frames into folders corresponding to scene. 
    """


    assert video_file_path.is_file()

    output_file = output_cuts_path / video_file_path.stem

    if not output_file.exists():

        model = TransNetV2(model_dir=model_path)
        video_frames, single_frame_predictions, all_frame_predictions = \
            model.predict_video(video_file_path)

        scenes = model.predictions_to_scenes(single_frame_predictions)

        with open(output_file, 'w') as file:
            np.savetxt(file, scenes, fmt="%d")

    else:
        print("Already found cuts...")


"""
import torch
from transnetv2_pytorch import TransNetV2

model = TransNetV2()
state_dict = torch.load("transnetv2-pytorch-weights.pth")
model.load_state_dict(state_dict)
model.eval().cuda()

with torch.no_grad():
    # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
    input_video = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)
    single_frame_pred, all_frame_pred = model(input_video.cuda())
    
    single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
    all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()

"""
#python transition_cuts.py 
#get video cuts
#calculate to fps
#take those frames and copy them to seperate folder 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file_path', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/kriss/datasets/videos/vid.mp4',
                        help='folder for downloading videos')
    parser.add_argument('--images_path', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/kriss/datasets/images',
                        help='folder for preprocessed images')
    parser.add_argument('--model_path', type=str,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/data/dev/TransNetV2/inference/transnetv2-weights',
                        help='transiton detection model weights')
    parser.add_argument('--output_cuts_path', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/kriss/datasets/cuts',
                        help='Where to store list of cuts for specific video')
    parser.add_argument('--output_folders_path', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/kriss/datasets/scenes',
                        help='Where to store the new folders of images')

    args = parser.parse_args()
    main(**args.__dict__)