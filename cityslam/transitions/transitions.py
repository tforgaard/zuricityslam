#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import ffmpeg
import torch
from transnetv2_pytorch import TransNetV2

"""
Adapted code from https://github.com/soCzech/TransNetV2/
"""

def predict_raw(frames: np.ndarray, model):
    with torch.no_grad():
        video_torch = torch.from_numpy(frames)
        single_frame_pred, all_frame_pred = model(video_torch.cuda())

        single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
        all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()


        return single_frame_pred, all_frame_pred

def predict_frames(frames: np.ndarray, model):
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

def main(video_file_path, model_path, output):
    assert video_file_path.is_file()

    output_file = output / video_file_path.stem

    if not output_file.exists():
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file_path', type=Path,
                        default='/cluster/home/ksteinsland/zuricityslam/base/kriss/datasets/videos/🇨🇭 Zürich Switzerland Walk 4K 🌁 4K Walking Tour ☁️ (Cloudy Day) 🇨🇭 [W25QdyiFnh0].mp4',
                        help='path to video')
    parser.add_argument('--model_path', type=str,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/data/dev/TransNetV2/inference/transnetv2-weights',
                        help='path to transiton detection model weights')
    parser.add_argument('--output', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/kriss/datasets/cuts',
                        help='where to store list of cuts for specific video')

    args = parser.parse_args()
    main(**args.__dict__)