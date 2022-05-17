from pathlib import Path
import numpy as np
import argparse

def create_img_list(cuts_path, images_dir, output, overlap=25, fps=2):

    for cut_file in Path(cuts_path).glob("*_cropped.txt"):

        print(cut_file)

        video_id = cut_file.name.split("_transitions")[0]
        image_folder = Path(images_dir) / video_id

        if not image_folder.exists():
            print(f"could not find {image_folder}, skipping!")
            continue

        Path(output).mkdir(exist_ok=True, parents=True)

        images = []
        globs = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
        for g in globs:
            images += list(image_folder.glob(g))
        images = sorted(images)

        with open(cut_file, 'r') as file:
            scenes = np.loadtxt(file, dtype=int)
            last_tranistion = 1
            for part, (scene_start, scene_end, transition) in enumerate(scenes):
                
                # add our images
                start_ind = scene_start
                stop_ind = scene_end
                
                if part != 0 and not last_tranistion:
                    # add last 25 
                    start_ind = max(scene_start - overlap, 0)

                if part+2 != len(scenes) and not transition:
                    # add next 25 images                        
                    stop_ind = min(scene_end + overlap, len(images) - 2)
                
                image_list_file = Path(output) / video_id / f"part{part}_images.txt"
                image_list_file.parent.mkdir(parents=True, exist_ok=True)
                with open(image_list_file, 'w+') as out_file:

                    for i in range(start_ind, stop_ind + 1):
                        out_file.write(str(images[i].relative_to(images_dir)) + "\n")

                last_tranistion = transition

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuts_path', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/transitions_cropped',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--images_dir', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images',
                        help='Path to the partioning of the datasets, default: %(default)s')
    parser.add_argument('--output', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/image_splits_new',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--overlap', type=int, default=25)
    parser.add_argument('--fps', type=int, default=2)
    args = parser.parse_args()

    create_img_list(**args.__dict__)
