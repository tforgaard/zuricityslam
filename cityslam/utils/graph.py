from pathlib import Path

from natsort import natsorted
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from hloc.utils import viz_3d
import pycolmap
import warnings


from cityslam.utils.parsers import model_path_2_name, model_name_2_path, get_model_base, find_models

def find_graphs(models, graph_dir):

    model_folders = find_models(models)
    model_names = [model_path_2_name(model) for model in model_folders]

    model_transforms = [p for p in Path(graph_dir).glob("**/trans_*")]

    transform_edges = [parse_merge_name(tf_name) for tf_name in model_transforms]
    transform_edges = [te for te in transform_edges if te is not None]

    G = nx.DiGraph()
    
    for (tf_u, tf_v), tf in transform_edges:
        G.add_edge(tf_u, tf_v, transform=tf)
    
    for name, folder in zip(model_names, model_folders):
        if name in G:
            G.add_node(name, model=folder)

    return G

def create_graph_from_model(name):

    G = nx.DiGraph()
    G.add_node(name, model=model_name_2_path(name))
    return G


def get_tf_filter_view(G):

    def transform_filter(model_1, model_2):
        if not G.has_edge(model_1, model_2):
            return False
        if G[model_1][model_2]['transform'] is None:
            return False
        return True

    return nx.subgraph_view(G, filter_edge=transform_filter)

def get_graphs(super_graph):
    '''Returns a list of disjoint graphs, sorted by largest graph first'''
    
    graphs = []

    super_graph_filtered = get_tf_filter_view(super_graph)

    for sub_graph_nodes in nx.connected_components(nx.to_undirected(super_graph_filtered)):
        graphs.append(super_graph.subgraph(sub_graph_nodes).copy())

    return sorted(graphs, key=len, reverse=True)


def get_tf(G, parent, child):
    if G.has_edge(parent, child):
        transform = G[parent][child].get('transform')
        # print("not inverse")
            
    elif G.has_edge(child, parent):
        transform = G[child][parent].get('transform')
        # print("inverse!")

    else:
        print("Edge does not exist!")
        return None

    if transform is None:
        print("could not find tf!")
    
    return transform

def transform_models(models, outputs, graph, base_node=None, visualize=False, save=True):
    if base_node is None:
        base_node = natsorted(list(graph.nodes))[0]

    outputs = outputs / base_node

    G = graph.copy()
    for node in G.nodes:
            G.nodes[node]['visited'] = False
    G.nodes[base_node]['visited'] = True

    b = pycolmap.Reconstruction(models / G.nodes[base_node]["model"])

    if save:
        (outputs/ "ply").mkdir(exist_ok=True,parents=True)
        b.export_PLY(outputs / "ply" / f"{base_node}.ply")
        
        model_dir = outputs / "models" / base_node
        model_dir.mkdir(exist_ok=True,parents=True)
        b.write(model_dir)

    if visualize:
        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(fig, b, color=rand_color(), name=base_node, points=False, cs=0.2)

    for parent, child in nx.bfs_edges(G.to_undirected(as_view=True), base_node, reverse=False):
        if not G.nodes[child]['visited']:
            G.nodes[child]['visited'] = True
            G.nodes[child]['parent'] = parent
            m = pycolmap.Reconstruction(models / G.nodes[child]["model"])
            
            debug_str = f"\n{parent} --> {child}"
            
            r_child = child
            while True:
                if G.nodes[r_child].get('parent') is None:
                    # print("reached base node")
                    break

                else:
                    r_parent = G.nodes[r_child]['parent']

                tf = get_tf(G, r_parent, r_child)
                if tf is None:
                    print("error: tf not found!")
                    break

                m.transform(tf)

                debug_str += f"\n\t{r_parent} <-- {r_child}"

                r_child = r_parent
            
            # print(debug_str)
            
            if save:
                m.export_PLY(outputs/ "ply" / f"{child}.ply")

                model_dir = outputs / "models" / child
                model_dir.mkdir(exist_ok=True)
                m.write(model_dir)
            
            if visualize:
                viz_3d.plot_reconstruction(fig, m, color=rand_color(), name=child, points=False, cs=0.2)
    
    if visualize:
        fig.show()

def transform_exists(graph, model_1, model_2, include_none=True):
    if not graph.has_edge(model_1, model_2):
        return False
    if graph[model_1][model_2]['transform'] is None and not include_none:
        return False
    # Returns true even if transform is None!
    print(f"transform already exists {model_1}, {model_2}")
    return True

def rand_color():
        return f'rgba({np.random.randint(0,256)},{np.random.randint(42,98)},{np.random.randint(40,90)},0.2)'

def rand_color_plt():
        return tuple(np.random.uniform(size=3))

def load_transform(tf_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = np.loadtxt(tf_path, delimiter=",")
    if len(matrix) == 0:
        # print("no transform")
        return None
    try:
        tf = pycolmap.SimilarityTransform3(matrix).inverse()
    except ValueError as e:
        # print("no transform")
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

    elif len(s) == 6:
        if "model" in s[2]:
            #vid1, p1, m, #, vid2, p2 = s
            name1 = "__".join(s[:4])
            name2 = "__".join(s[4:])
        elif "model" in s[-2]:
            #vid1, p1, vid2, p2, m, # = s
            name1 = "__".join(s[:2])
            name2 = "__".join(s[2:])

    elif len(s) == 8:
        #vid1, p1, m, #, vid2, p2, m, # = s
        name1 = "__".join(s[:4])
        name2 = "__".join(s[4:])
    else:
        return None
    return (name1, name2), load_transform(tf_path)

def draw_graphs(graphs):


    for graph in graphs:
        if graph.number_of_nodes() > 1:
            pos = nx.spring_layout(graph, k=1.0)
            plt.figure()
            
            #nx.draw(graph, with_labels=True)
            nx.draw(get_tf_filter_view(graph), pos)
            labels = {}
            for node in graph.nodes:
                labels[node] = "/".join(model_name_2_path(node).parts[1:])
            nx.draw_networkx_labels(graph, pos, labels)
            # plt.xlabel(model_name_2_path(list(graph.nodes)[0]))
            models = list(set([model_name_2_path(node).parts[0] for node in graph.nodes]))
            plt.title(" ".join(models))
            plt.draw()

def draw_super(G, models):

    groups = set([get_model_base(models, model_name_2_path(node)).name for node in G.nodes])
    
    pos = nx.spring_layout(G, k=1.0)
    nx.draw_networkx_edges(get_tf_filter_view(G), pos, width=1.0, alpha=0.5)
    for group in groups:
        group_nodes = [model_name for model_name in G.nodes if group in model_name]
        group_nodes = [node for node in group_nodes for (tf_u, tf_v) in G.edges if (node == tf_u or node == tf_v) and G[tf_u][tf_v]['transform'] is not None]
        nx.draw_networkx_nodes(G, pos, nodelist=group_nodes, node_color=[rand_color_plt()])
