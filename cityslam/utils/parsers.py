from pathlib import Path

import pycolmap
from natsort import natsorted

def find_models(models_dir, models_mask=None):
    
    # Recursively search for all models and make the model_folder paths relative to models_dir
    model_folders = [p.parent.relative_to(
        models_dir) for p in Path(models_dir).glob("**/images.bin")]

    # Optionally only include specific models
    if models_mask is not None:
        if isinstance(models_mask, str):
            models_mask = [models_mask]
        model_folders = [
            model_folder for model_folder in model_folders for model_mask in models_mask if model_mask in model_folder.parts]

    # Remove redundant folders
    # If we have reconstructions in the PATH/models/[0-9] folders
    # remove the reconstruction in PATH, as this is redundant
    remove_folders = list(set([model_folder.parent.parent for model_folder in model_folders if model_folder.name.isdigit()]))

    return natsorted([model_folder for model_folder in model_folders if model_folder not in remove_folders])


def get_images_from_recon(sfm_model):
    """Get a sorted list of images in a reconstruction"""
    # NB! This will most likely be a SUBSET of all the images in a folder like images/gTHMvU3XHBk

    if isinstance(sfm_model, (str, Path)):
        sfm_model = pycolmap.Reconstruction(sfm_model)

    img_list = [img.name for img in sfm_model.images.values()]

    return sorted(img_list)


def get_images(image_path, subfolder=None):
    globs = ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG']
    image_list = []
    
    if subfolder is not None:
        image_path = image_path / subfolder
    
    for g in globs:
        image_list += list(Path(image_path).glob(g))

    image_list = ["/".join(img.parts[-2:]) for img in image_list]
        
    image_list = sorted(list(image_list))
    return image_list


def model_path_2_name(model_path):
    return str(model_path).replace("/", "__")


def model_name_2_path(model_name):
    # backwards comp
    model_name = str(model_name).replace('__sfm_sp+sg', '')

    return Path(str(model_name).replace("__", "/"))


def get_model_base(model_folder, relative_model_path):
    return Path(model_folder) / Path(relative_model_path).parts[0]

def sequential_models(first_model, second_model, direction='forward'):
    seq_n_first = int(first_model.parts[1].split("part")[-1])
    seq_n_second = int(second_model.parts[1].split("part")[-1])

    if not (first_model.parts[0] == second_model.parts[0]):
        return False

    if direction=='forward':
        if not (seq_n_first + 1 == seq_n_second):
            return False
    elif direction=='backward':
        if not (seq_n_first - 1 == seq_n_second):
            return False
    elif direction is None or direction =='none':
        if not (seq_n_first + 1 == seq_n_second or seq_n_first - 1 == seq_n_second):
            return False
    else:
        raise KeyError(direction)
        
    return True