from pathlib import Path
import argparse

import yt_dlp


def main(output, video_ids, format, overwrite=False):
    output.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        'format': format,
        'paths': {'home': f'{output}'},  # home is download directory...
        'output': {'home': '%(id)s'},  # not sure about this
        'retries': 10
    }

    if overwrite:
        ydl_opts['overwrites'] = True

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for video in video_ids:
            ydl.download(['https://www.youtube.com/watch?v=' + video])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('video_ids', nargs="+",
                        help='list of video ids to download')
    parser.add_argument('--output', type=Path,
                        default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/videos',
                        help='folder for downloading videos')
    parser.add_argument('--format', type=str, default='wv',
                        help='Download format to fetch, default: %(default)s (select best video)',
                        choices=['bv', 'wv'])  # and more!
    parser.add_argument("--overwrite",
                        help="Overwrite cached queries",
                        action="store_true")
    args = parser.parse_args()

    main(**args.__dict__)
