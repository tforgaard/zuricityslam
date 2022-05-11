from pathlib import Path
import argparse
import os

import ffmpeg

from .. import logger


def frame_capture(video_path, images_path, prefix="", fps=2, start='00:00:00', duration='00:00:00'):
    if prefix:
        prefix = prefix + "_"

    input_opts = {'ss': start}
    if duration != '00:00:00':
        input_opts = {'t': duration}

    try:
        ffmpeg.input(video_path, **input_opts) \
            .filter('fps', fps=f'{fps}') \
            .output(str(images_path / f'{prefix}img_fps{fps}_%05d.jpg'), start_number=0) \
            .overwrite_output() \
            .run(quiet=True)
    except ffmpeg.Error as e:
        print('stdout:', e.stdout.decode('utf8'))
        print('stderr:', e.stderr.decode('utf8'))
        raise e


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
