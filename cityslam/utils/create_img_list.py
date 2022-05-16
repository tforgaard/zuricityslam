from pathlib import Path
import numpy as np

def create_img_list(cuts_path, images_dir, output, overlap=25, fps=2):

    for cut_file in Path(cuts_path).glob("*_cropped.txt"):

        print(cut_file)

        video_id = cut_file.name.split("_transitions")[0]
        image_folder = Path(images_dir) / video_id

        assert image_folder.exists(), image_folder 

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
                
                image_list_file = Path(output) / f"{video_id}_part{part}_images.txt"
                with open(image_list_file, 'w+') as out_file:

                    for i in range(start_ind, stop_ind + 1):
                        out_file.write(str(images[i].relative_to(images_dir)) + "\n")


                last_tranistion = transition

if __name__ == "__main__":

    create_img_list("./base/datasets/transitions", "./base/datasets/images", "./base/datasets/image_splits")