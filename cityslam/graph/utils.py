import networkx as nx

import networkx as nx
import matplotlib.pyplot as plt


from pathlib import Path
from hloc.utils import viz_3d
import pycolmap
import numpy as np

from cityslam.utils.parsers import model_path_2_name, model_name_2_path, get_model_base, find_models

def find_graphs(models, graph_dir):

    model_folders = find_models(models)
    
    model_names = ["__".join(model_f.parts[:2]) for model_f in model_folders]
    model_names = np.unique(model_names)

    # model_bases = set([get_model_base(models, model_folder).name for model_folder in model_folders])
    # model_indexes = [i for model_name in model_names for i, model_base in enumerate(model_bases) if model_base in model_name]

    model_transforms = [p for p in Path(graph_dir).glob("**/trans_*")]
    # model_transform_names = [p.name for p in model_transforms]

    transform_edges = [parse_merge_name(tf_name) for tf_name in model_transforms]
    transform_edges = [te for te in transform_edges if te is not None]


    G = nx.DiGraph()
    
    for (tf_u, tf_v), tf in transform_edges:
        
        if tf is None: # We have tried to merge the two models without success, add nodes
            G.add_node(name, model=model_name_2_path(name))
            G.add_node(name, model=model_name_2_path(name))
        else:
            G.add_edge(tf_u, tf_v, transform=tf)
    
    for name, folder in zip(model_names, model_folders):
        if name in G:
            G.add_node(name, model=folder)

    return G

def create_graph_from_model(name):

    G = nx.DiGraph()
    G.add_node(name, model=model_name_2_path(name))
    return G

def get_graphs(super_graph):
    '''Returns a list of disjoint graphs, sorted by largest graph first'''
    graphs = []

    for sub_graph_nodes in nx.connected_components(nx.to_undirected(super_graph)):
        graphs.append(super_graph.subgraph(sub_graph_nodes).copy())

    return sorted(graphs, key=len, reverse=True)


def transform_models(models, outputs, graph, base_node=None):
    # Set this to the first node in the chain!
    if base_node is None:
        base_node = graph.nodes[0]
    
    G_t = graph.copy()
    for node in G_t.nodes:
            G_t.nodes[node]['visited'] = False

    G_t.nodes[base_node]['visited'] = True
    fig = viz_3d.init_figure()
    b = pycolmap.Reconstruction(models / G_t.nodes[base_node]["model"])
    viz_3d.plot_reconstruction(fig, b, color=rand_color(), name=base_node, points=True, cs=0.2)
    b.export_PLY(outputs/ f"{base_node}.ply")
    b.export_NVM(outputs/ f"{base_node}.nvm", skip_distortion=False)
    for parent, child, _ in nx.edge_bfs(G, base_node, orientation='original'):
        if not G_t.nodes[child]['visited']:
            G_t.nodes[child]['visited'] = True
            m = pycolmap.Reconstruction(models / G.nodes[child]["model"])
            m.transform(G_t[parent][child]['transform'])
            print(child)
            print(parent)
            for reverse_parent, reverse_child, _ in nx.edge_bfs(G_t, parent, orientation='reverse'):

                    m.transform(G_t[reverse_parent][reverse_child]['transform'])

            viz_3d.plot_reconstruction(fig, m, color=rand_color(), name=child, points=True, cs=0.2)
            m.export_PLY(outputs/ f"{child}.ply")


def rand_color():
        return f'rgba({np.random.randint(0,256)},{np.random.randint(42,98)},{np.random.randint(40,90)},0.2)'

def rand_color_plt():
        return tuple(np.random.uniform(size=3))

def load_transform(tf_path):
    try:
        tf = pycolmap.SimilarityTransform3(np.loadtxt(tf_path, delimiter=",")).inverse()
    except ValueError as e:
        print("no transform")
        return None

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

def draw_graphs(graphs):

    import matplotlib.pyplot as plt
    for graph in graphs:
        if graph.number_of_nodes() > 1:
            plt.figure()
            nx.draw(graph)


def draw_super(G, models, model_names):

    groups = set([get_model_base(models, model_name_2_path(model_name)).name for model_name in model_names])
    
    pos = nx.spring_layout(G, k=1.0)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    for group in groups:
        group_nodes = [model_name for model_name in model_names if group in model_name]
        group_nodes = [node for node in group_nodes for (tf_u, tf_v), tf in G.edges if node == tf_u or node == tf_v]
        nx.draw_networkx_nodes(G, pos, nodelist=group_nodes, node_color=[rand_color_plt()])

