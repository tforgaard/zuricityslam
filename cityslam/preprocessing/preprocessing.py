from pathlib import Path
import argparse
import os

import ffmpeg

from .. import logger
#multiprocessing frame splitting
import cv2
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time

def process_video_multiprocessing(group_number, frame_jump_unit, interval, file_name, images_path, prefix, img_start_name, pad):
    # Read video file
    cap = cv2.VideoCapture(file_name)
    frameNr = frame_jump_unit * group_number
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
    img_name = img_start_name * group_number
    count = 0
    while count < frame_jump_unit:
        ret, frame = cap.read()

        if not ret:
            break

        if (count % interval == 0):
            image_path = os.path.join(images_path, prefix + '_' + str(img_name).zfill(pad) + '.jpg')
            cv2.imwrite(image_path, frame)
            img_name += 1
        count += 1

    cap.release()

def frame_capture(video_path, images_path, prefix="", fps=2, start='00:00:00', duration='00:00:00'):
    if prefix:
        prefix = prefix + "_"

    file_name = str(video_path)
    print(file_name)
    cap = cv2.VideoCapture(file_name)
    if cap is None or not cap.isOpened():
       print('Warning: unable to open video source: ', file_name)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    #num_processes = mp.cpu_count()
    num_processes = 10
    frame_jump_unit =  int(frame_count / num_processes)
    interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
    img_start_name = int(frame_jump_unit / interval) + 1

    start_time = time.time()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(partial(process_video_multiprocessing,
                             frame_jump_unit = frame_jump_unit,
                             interval = interval,
                             file_name = file_name,
                             images_path = images_path,
                             prefix = prefix, 
                             img_start_name = img_start_name,
                             pad = len(str(frame_count))), range(num_processes))

    end_time = time.time()

    total_processing_time = end_time - start_time
    print("Time taken: {}".format(total_processing_time))

def main(videos, output, video_ids=None, overwrite=False, fps=2, start='00:00:00', duration='00:00:00'):

    image_dirs = []
    for file in os.listdir(videos):
        if file.endswith((".mp4", "webm")):
            video_id = file[file.find("[")+1:file.find("]")]
            
            # skip videoes not in video_ids
            if video_ids is not None and video_id not in video_ids:
                continue

            logger.info(
                f"extracting frames from video: {file} using fps: {fps}")
            video_path = videos / file  # video_path
            images_path = output / f"{video_id}"
            image_dirs.append(images_path)
            images_path.mkdir(parents=True, exist_ok=True)
            if list(images_path.glob(f"*_fps{fps}_*.jpg")) and not overwrite:
                logger.info(f"frames already extracted for video: {file}")
                continue
            frame_capture(video_path, images_path, prefix=video_id,
                          fps=fps, start=start, duration=duration)

    return image_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/videos',
                        help='folder for downloaded videos')
    parser.add_argument('--output', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images',
                        help='folder for preprocessed images')
    parser.add_argument('--video_ids', type=int, default=None,
                        help='Video ids to preprocess, defaults to do everyone')
    parser.add_argument("--overwrite",
                        help="Overwrite cached queries",
                        action="store_true")
    parser.add_argument('--fps', type=int, default='2',
                        help='fps to use in image splitting')
    parser.add_argument("--start", type=str, default='00:00:00',
                        help="Start of video to split into images, %(default)s")
    parser.add_argument("--duration", type=str, default='00:00:00',
                        help="How many minutes of the video to convert, %(default)s")
    args = parser.parse_args()
    image_dirs = main(**args.__dict__)
    print(" ".join(image_dirs))
