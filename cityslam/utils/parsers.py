from pathlib import Path

import pycolmap

def get_images_from_recon(sfm_model):
    """Get a sorted list of images in a reconstruction"""
    # NB! This will most likely be a SUBSET of all the images in a folder like images/gTHMvU3XHBk

    if isinstance(sfm_model, (str, Path)):
        sfm_model = pycolmap.Reconstruction(sfm_model)
    
    img_list = [img.name for img in sfm_model.images.values()]
    
    return sorted(img_list)


def model_path_2_name(model_path:str):
    return str(model_path).replace("/","__")

def model_name_2_path(model_path):
    return Path(str(model_path).replace("__","/"))

def get_model_base(model_folder, relative_model_path):
    return Path(model_folder) / Path(relative_model_path).parts[0]