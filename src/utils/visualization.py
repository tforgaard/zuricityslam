import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from matplotlib import cm
import numpy as np
from pathlib import Path
import pycolmap
# from multiprocessing import Pool, cpu_count, Process
# import functools

from hloc.utils.viz import (
        plot_video_frames, make_video, plot_keypoints, add_text)
from hloc.utils.io import read_image


def plot_video_frames(sfm_data, save_dir='./', titles=None, cmaps='gray', dpi=100):
    """Plot a set of images sequentially.
    Args:
        imgs: a list of NumPy or PyTorch images, RGB (H, W, 3) or mono (H, W).
        titles: a list of strings, as titles for each image.
        cmaps: colormaps for monochrome images.
        adaptive: whether the figure size should fit the image aspect ratios.
    """

    for i, (image, name, keypoints, color, text) in enumerate(sfm_data):
        if type(cmaps) == str:
            cmap = plt.get_cmap(cmaps)
        else:
            cmap = plt.get_cmap[cmaps[i]]
        if i == 0:
            ratio = image.shape[1] / image.shape[0] # W / H

            figsize = [ratio*4.5, 4.5]
            fig = plt.figure(
                figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(1, 1, 1)

            ax.get_yaxis().set_ticks([])
            ax.get_xaxis().set_ticks([])
            ax.set_axis_off()

            img_plt = ax.imshow(image, cmap=cmap)

            for spine in ax.spines.values():  # remove frame
                spine.set_visible(False)
        
        else:
            for scatter in scatters:
                scatter.remove()
            t1.remove()
            t2.remove()
            img_plt.set_data(image)
            img_plt.set_cmap(cmap)

        if titles:
            ax.set_title(titles[i])

        scatters = plot_keypoints([keypoints], colors=[color], ps=4)
        t1 = add_text(0, text)
        t2 = add_text(0, name, pos=(0.01, 0.01), fs=5, lcolor=None, va='bottom')
        plt.draw()
        fig.savefig(Path(save_dir) / f"{i}.png")


def make_video(video_frames, output, fps=2):

    images = [img for img in video_frames.glob("*.png")]
    frame = cv2.imread(str(video_frames / images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(str(output), 0, fps, (width,height))

    for image in images:
        video.write(cv2.imread(str(video_frames / image)))

    cv2.destroyAllWindows()
    video.release()


def visualize_sfm_2d_video(reconstruction, image_dir, output, color_by='visibility',
                     image_list=[], start=None, stop=None, dpi=75):
    assert image_dir.exists()
    if not isinstance(reconstruction, pycolmap.Reconstruction):
        reconstruction = pycolmap.Reconstruction(reconstruction)

    if not image_list:
        image_list = reconstruction.reg_image_ids()
        if not start:
            start = 0
        if not stop:
            stop = len(image_list)

        image_list = image_list[start:stop]

    output = Path(output)
    output.mkdir(exist_ok=True)
    video_output = output / 'test_video.avi'
    video_frames_output = output / 'video_frames_tmp'
    video_frames_output.mkdir(exist_ok=True)

    def sfm_data(image_list, reconstruction, color_by):
        n = 0
        for i in image_list:
            image = reconstruction.images[i]
            keypoints = np.array([p.xy for p in image.points2D])
            visible = np.array([p.has_point3D() for p in image.points2D])

            if color_by == 'visibility':
                color = [(0, 0, 1) if v else (1, 0, 0) for v in visible]
                text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
            elif color_by == 'track_length':
                tl = np.array([reconstruction.points3D[p.point3D_id].track.length()
                            if p.has_point3D() else 1 for p in image.points2D])
                max_, med_ = np.max(tl), np.median(tl[tl > 1])
                tl = np.log(tl)
                color = cm.jet(tl / tl.max()).tolist()
                text = f'max/median track length: {max_}/{med_}'
            elif color_by == 'depth':
                p3ids = [p.point3D_id for p in image.points2D if p.has_point3D()]
                z = np.array([image.transform_to_image(
                    reconstruction.points3D[j].xyz)[-1] for j in p3ids])
                z -= z.min()
                color = cm.jet(z / np.percentile(z, 99.9))
                text = f'visible: {np.count_nonzero(visible)}/{len(visible)}'
                keypoints = keypoints[visible]
            else:
                raise NotImplementedError(f'Coloring not implemented: {color_by}.')

            name = image.name
            yield read_image(image_dir / name), name, keypoints, color, text
    
    plot_video_frames(sfm_data(image_list, reconstruction, color_by), video_frames_output, dpi=dpi)
    make_video(video_frames_output, video_output)


if __name__ == '__main__':
    base = Path('/cluster/project/infk/courses/252-0579-00L/group07')
    reconstruction = base /'outputs/loop_walk_zurich_sequential+retrieval_fps2/sfm_superpoint+superglue/colmap/0'
    image_dir = base / 'datasets/loop_walk_zurich/images-fps2/'
    output = base / 'outputs/loop_walk_zurich_sequential+retrieval_fps2/sfm_superpoint+superglue/colmap/0'
    #stop=30
    visualize_sfm_2d_video(reconstruction, image_dir, output) #, stop=stop)