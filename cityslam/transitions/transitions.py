#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import ffmpeg
import torch
from .transnetv2_pytorch import TransNetV2

"""
Adapted code from https://github.com/soCzech/TransNetV2/
"""

def predict_raw(frames: np.ndarray, model, device):
    with torch.no_grad():
        video_torch = torch.from_numpy(frames)
        single_frame_pred, all_frame_pred = model(video_torch.to(device))

        single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
        all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()


        return single_frame_pred, all_frame_pred

def predict_frames(frames: np.ndarray, model, device):
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
            single_frame_pred, all_frames_pred = predict_raw(inp, model, device)
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

def add_max_min_cuts(video_file_path, max_scene_length, min_scene_length, cut_file, fps=2):
    probe = ffmpeg.probe(video_file_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    video_fps = float(video_info['r_frame_rate'].split('/')[0])
    
    # Sometimes the frame rate is given as frames per 1000 seconds?
    if video_fps / 1000 > 1.0:
        video_fps = video_fps / 1000
    
    print(f"video fps: {video_fps}")
    max_frames = int(video_fps*max_scene_length)
    min_frames = int(video_fps*min_scene_length)

    fps_ratio = video_fps / fps

    """
    we drop too short scenes,
    for the too long scenes we want to divde them up: 
    [[scene[0], scene[0] + max_frames - 1],
        [scene[0] + max_frames, scene[0] + 2*max_frames - 1],
    ...
        [scene[0] + (whole_maxes - 1)*max_frames, scene[0] + (whole_maxes)*max_frames - 1],
        [scene[0] + (whole_maxes)*max_frames, scene[1]]
    """    

    new_scenes = []
    with open(cut_file, 'r') as file:
        scenes = np.loadtxt(file, dtype=int)
        for i, (scene_start, scene_end) in enumerate(scenes):
            scene_length = scene_end - scene_start
            whole_maxes = int(scene_length // max_frames)
            rest_maxes = int(scene_length % max_frames)

            if scene_length > min_frames:
                
                for j in range(whole_maxes):
                    transition = j == 0 and i != 0
                    new_scenes.append([int((scene_start + j*max_frames) // fps_ratio), int((scene_start + (j+1)*max_frames - 1) // fps_ratio), transition])

                if rest_maxes >= min_frames:
                    transition = whole_maxes == 0
                    new_scenes.append([int((scene_start + (whole_maxes)*max_frames) // fps_ratio), int(scene_end // fps_ratio), transition])
        
    with open(f"{cut_file.parent / cut_file.stem}_cropped.txt", 'w') as file:
            np.savetxt(file, new_scenes, fmt="%d")


def main(videos_dir, video_ids, model_path, output, max_scene_length, min_scene_length, fps, threshold, overwrite=False):
    videos_dir = Path(videos_dir)
    assert videos_dir.exists(), videos_dir    

    model = TransNetV2()
    model_weights = Path(model_path) / "transnetv2-pytorch-weights.pth"
    assert model_weights.exists(), model_weights
    state_dict = torch.load(str(model_weights))
    model.load_state_dict(state_dict)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"using device {device}")
    model.eval().to(device)

    if video_ids is None:
        video_ids = [file[file.find("[")+1:file.find("]")] for file in videos_dir.iterdir()]

    for video_id in video_ids:

        video_file_path = next(videos_dir.glob(f"*{video_id}*"))
        if not video_file_path:
            print(f"could not find video with id {video_id}, skipping")
            continue

        output_file = Path(output) / f"{video_id}_transitions.txt"
        output_file.parent.mkdir(exist_ok=True, parents=True)
        
        if not output_file.exists() or overwrite:

            video_stream, err = ffmpeg.input(video_file_path).output(
                    "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
                ).run(capture_stdout=True, capture_stderr=True)

            video = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])

            print(f"Finding transitions for video {video_file_path.stem}")
            single_frame_pred, all_frames_pred = predict_frames(video, model, device)

            scenes = predictions_to_scenes(single_frame_pred, threshold)

            with open(output_file, 'w') as file:
                np.savetxt(file, scenes, fmt="%d")

        else:
            print(f"Already found transitions for video {video_file_path.stem}")

        add_max_min_cuts(video_file_path, max_scene_length, min_scene_length, output_file, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', type=Path,
                        default='/cluster/home/ksteinsland/zuricityslam/base/kriss/datasets/videos/🇨🇭 Zürich Switzerland Walk 4K 🌁 4K Walking Tour ☁️ (Cloudy Day) 🇨🇭 [W25QdyiFnh0].mp4',
                        help='path to videos')
    parser.add_argument('--video_ids', type=str, nargs="+",
                            default=None,
                        help='video_id')
    parser.add_argument('--model_path', type=str,
                        default='./', #'/cluster/project/infk/courses/252-0579-00L/group07/data/dev/TransNetV2/inference/transnetv2-weights',
                        help='path to transiton detection model weights')
    parser.add_argument('--output', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/kriss/datasets/cuts',
                        help='where to store list of cuts for specific video')
    parser.add_argument('--max_scene_length', type=int,
                        default=5*60,
                        help='number of seconds of max scene length')
    parser.add_argument('--min_scene_length', type=int,
                        default=30,
                        help='number of seconds of min scene length')
    parser.add_argument('--fps', type=int,
                        default=2,
                        help='fps of output cropped file')
    parser.add_argument('--threshold', type=float,
                        default=0.5,
                        help='transition threshold')
    parser.add_argument('--overwrite', action="store_true")

    args = parser.parse_args()
    main(**args.__dict__)