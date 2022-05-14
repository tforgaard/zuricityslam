#!/usr/bin/env python

#from transnetv2 import TransNetV2
import argparse
from pathlib import Path
import numpy as np
#from transnetv2 import TransNetV2
import ffmpeg

import torch
from transnetv2_pytorch import TransNetV2

"""
def find_transitions(video_file_path, model_path, output_cuts_path):
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


def predict_raw(frames: np.ndarray, model):
    """
    assert len(frames.shape) == 5 and frames.shape[2:] == self._input_size, \
        "[TransNetV2] Input shape must be [batch, frames, height, width, 3]."
    frames = tf.cast(frames, tf.float32)

    logits, dict_ = self._model(frames)
    single_frame_pred = tf.sigmoid(logits)
    all_frames_pred = tf.sigmoid(dict_["many_hot"])
    """
    with torch.no_grad():
        # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels
        #input_video = torch.zeros(1, 100, 27, 48, 3, dtype=torch.uint8)
        video_torch = torch.from_numpy(frames)
        #video_torch = video_torch[None, :]
        #print(video_torch.shape)
        single_frame_pred, all_frame_pred = model(video_torch.cuda())

        single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
        all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()


        return single_frame_pred, all_frame_pred

def predict_frames(frames: np.ndarray, model):
        #assert len(frames.shape) == 4 and frames.shape[1:] == self._input_size, \
        #    "[TransNetV2] Input shape must be [frames, height, width, 3]."

        def input_iterator():
            # return windows of size 100 where the first/last 25 frames are from the previous/next batch
            # the first and last window must be padded by copies of the first and last frame of the video
            no_padded_frames_start = 25
            no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

            start_frame = np.expand_dims(frames[0], 0)
            end_frame = np.expand_dims(frames[-1], 0)
            padded_inputs = np.concatenate(
                [start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
            )

            ptr = 0
            while ptr + 100 <= len(padded_inputs):
                out = padded_inputs[ptr:ptr + 100]
                ptr += 50
                yield out[np.newaxis]

        predictions = []

        for inp in input_iterator():
            single_frame_pred, all_frames_pred = predict_raw(inp, model)
            predictions.append((single_frame_pred[0, 25:75, 0],
                                all_frames_pred[0, 25:75, 0]))

            print("\r[TransNetV2] Processing video frames {}/{}".format(
                min(len(predictions) * 50, len(frames)), len(frames)
            ), end="")
        print("")

        single_frame_pred = np.concatenate([single_ for single_, all_ in predictions])
        all_frames_pred = np.concatenate([all_ for single_, all_ in predictions])

        return single_frame_pred[:len(frames)], all_frames_pred[:len(frames)]  # remove extra padded frames


"""
def find_transitions(video_file_path, images_path, model_path, output_cuts_path, output_folders_path):
    model = TransNetV2()
    state_dict = torch.load("transnetv2-pytorch-weights.pth")
    model.load_state_dict(state_dict)
    model.eval().cuda()

    video_stream, err = ffmpeg.input(video_file_path).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)

    video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])

    single_frame_pred, all_frames_pred = predict_frames(video, model)

    return single_frame_pred, all_frames_pred

"""



def split_videos(video_file_path, output_folders_path, cuts_path):
    """
    out, err = (
        ffmpeg
        .input(video_file_path)
        .filter('select', 'between(10,20)')
        .output('output.mp4')
        .run()
    )
    
    split = (
        ffmpeg
        .input(video_file_path)
        .filter_multi_output('split')
    )
    ffmpeg.output(split[0], 'output1.mp4').run()
    ffmpeg.output(split[1], 'output2.mp4').run()
    """

"""
def predict_video(video_fn: str):
        try:
            import ffmpeg
        except ModuleNotFoundError:
            raise ModuleNotFoundError("For `predict_video` function `ffmpeg` needs to be installed in order to extract "
                                      "individual frames from video file. Install `ffmpeg` command line tool and then "
                                      "install python wrapper by `pip install ffmpeg-python`.")

        print("[TransNetV2] Extracting frames from {}".format(video_fn))
        video_stream, err = ffmpeg.input(video_fn).output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
        ).run(capture_stdout=True, capture_stderr=True)

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])
        return video
"""

def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
        predictions = (predictions > threshold).astype(np.uint8)

        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])

        # just fix if all predictions are 1
        if len(scenes) == 0:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)

        return np.array(scenes, dtype=np.int32)

def main(video_file_path, images_path, model_path, output_cuts_path, output_folders_path):
    """
    Finds transitions in videos, divides frames into folders corresponding to scene. 
    """

    assert video_file_path.is_file()

    output_file = output_cuts_path / video_file_path.stem

    if not output_file.exists():

        #single_frame_pred, all_frames_pred = find_transitions(video_file_path, images_path, model_path, output_cuts_path, output_folders_path)
        #split_videos(video_file_path, output_folders_path, output_cuts_path)
        model = TransNetV2()
        state_dict = torch.load("transnetv2-pytorch-weights.pth")
        model.load_state_dict(state_dict)
        model.eval().cuda()

        video_stream, err = ffmpeg.input(video_file_path).output(
                "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
            ).run(capture_stdout=True, capture_stderr=True)

        video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])

        single_frame_pred, all_frames_pred = predict_frames(video, model)

        scenes = predictions_to_scenes(single_frame_pred)

        with open(output_file, 'w') as file:
            np.savetxt(file, scenes, fmt="%d")

    else:
        print(f"Already found cuts for video {video_file_path.stem}")


    


"""


split = ffmpeg.input(video_file_path)
split = ffmpeg.filter_multi_output(split, 'split')

ffmpeg.output(split[0], 'output1.mp4')
ffmpeg.output(split[1], 'output2.mp4')


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
                        default='/cluster/home/ksteinsland/zuricityslam/base/kriss/datasets/videos/🇨🇭 Zürich Switzerland Walk 4K 🌁 4K Walking Tour ☁️ (Cloudy Day) 🇨🇭 [W25QdyiFnh0].mp4',
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