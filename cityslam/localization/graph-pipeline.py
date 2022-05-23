import networkx as nx
import matplotlib.pyplot as plt
import argparse

from pathlib import Path
from hloc.utils import viz_3d
import pycolmap
import numpy as np
from cityslam.utils.parsers import get_images_from_recon, model_path_2_name, model_name_2_path, get_model_base

def rand_color():
        return f'rgba({np.random.randint(0,256)},{np.random.randint(42,98)},{np.random.randint(40,90)},0.2)'


def load_transform(tf_path):
    with open(tf_path, "r") as f:
        tf = pycolmap.SimilarityTransform3(np.loadtxt(f,delimiter=",")).inverse()
    return tf

def parse_merge_name(tf_path):
    tf_name = tf_path.name
    suffix=".txt"
    if tf_name.endswith(suffix):
        tf_name = tf_name[:-len(suffix)]

    s =  tf_name.split("__")[1:]

    if len(s) == 2:
        name1, name2 = s
    
    elif len(s) == 4:
        # vid1, p1, vid2, p2 = s
        name1 = "__".join(s[:2])
        name2 = "__".join(s[2:])

    elif len(s) == 5:
        if "model" in s[2]:
            #vid1, p1, m1, vid2, p2 = s
            name1 = "__".join(s[:3])
            name2 = "__".join(s[3:])
        elif "model" in s[-1]:
            #vid1, p1, vid2, p2, m2 = s
            name1 = "__".join(s[:2])
            name2 = "__".join(s[2:])
    else:
        return None
    return (name1, name2), load_transform(tf_path)

def main(images, models, merge, outputs):

    outputs.mkdir(exist_ok=True, parents=True)

    model_folders = [p.parent for p in Path(models).glob("**/images.bin")]
    remove_folders = []
    for model_folder in model_folders:
        
        # If we have reconstructions in the PATH/models/[0-9] folders
        # Then we should remove the reconstruction in PATH, as this reconstruction
        # is the same as one of the ones in PATH/models/[0-9]
        if model_folder.name.isdigit():    
            rem_folder = model_folder.relative_to(models)
            remove_folders.append(rem_folder)

    # Make the model_folder paths relative to models_dir and remove redundant folders
    model_folders = [model_folder.relative_to(models) for model_folder in model_folders if model_folder not in remove_folders]
    model_names = ["__".join(model_f.parts[:2]) for model_f in model_folders]
    model_names = np.unique(model_names)

    model_transforms = [p for p in Path(merge).glob("**/trans_*")]
    # model_transform_names = [p.name for p in model_transforms]
    transform_edges = []
    transform_edges = [parse_merge_name(tf_name) for tf_name in model_transforms]
    transform_edges = [te for te in transform_edges if te is not None]

    #for i in transform_edges:
    #    print(i)
    
    G = nx.DiGraph()
    for name, folder in zip(model_names, model_folders):
        G.add_node(name, model=folder)

    for (tf_u, tf_v), tf in transform_edges:
        G.add_edge(tf_u, tf_v, transform=tf)

    nx.draw(G)
    plt.savefig("Graph.png", format="PNG")

    # Set this to the first node in the chain!
    #base_node = '2obsKLoZQdU__part15'
    base_node = '73IVzh0R-Lo__part0'
    fig = viz_3d.init_figure()
    b = pycolmap.Reconstruction(models / G.nodes[base_node]["model"])
    viz_3d.plot_reconstruction(fig, b, color=rand_color(), name=base_node, points=False, cs=0.2)
    b.export_PLY(outputs/ f"{base_node}.ply")

    for parent, child, _ in nx.edge_bfs(G, base_node, orientation='original'):
        
        m = pycolmap.Reconstruction(models / G.nodes[child]["model"])
        m.transform(G[parent][child]['transform'])
        
        for reverse_parent, reverse_child, _ in nx.edge_bfs(G, parent, orientation='reverse'):

                m.transform(G[reverse_parent][reverse_child]['transform'])

        viz_3d.plot_reconstruction(fig, m, color=rand_color(), name=child, points=False, cs=0.2)
        m.export_PLY(outputs/ f"{child}.ply")
    
    fig.write_image("reconstruction.png")
    fig.write_html("reconstruction.html")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/datasets/images',
                        help='Path to the dataset, default: %(default)s')
    parser.add_argument('--models', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/models-features',
                        help='Path to the model directory, default: %(default)s')
    parser.add_argument('--merge', type=Path, default='/cluster/project/infk/courses/252-0579-00L/group07/outputs/test/merge',
                        help='Path to the merge model directory, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/cluster/home/skalanan/graph',
                        help='Path to the output directory, default: %(default)s')                              
    args = parser.parse_args()
    
    # Run mapping
    model = main(**args.__dict__)
