import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.transform import Rotation as R
from ssl_neuron.utils import neighbors_to_adjacency


def connect_graph(adj_matrix, neighbors, features, verbose=False):
    """
    Check if graph consists of only one connected component. If not iterativly connect two points of two unconnected components with shortest distance between them.
    Args:
        adj_matrix: adjacency matrix of graph (N x N)
        neighbors: dict of neighbors per node
        features: features per node (N x D)
    """
    neighbors2 = {k: set(v) for k, v in neighbors.items()}
    G = nx.Graph(adj_matrix)
    num_comp = nx.number_connected_components(G)
    count = 1
    while num_comp > 1:
        components = {i: l for i, l in enumerate(list(nx.connected_components(G)))}
        components_ids = list(components.keys())
        for i, c_id in enumerate(components_ids):
            nodes = components[c_id]
            leaf_nodes = [n for n in nodes if len(neighbors2[n]) == 1]

            if len(leaf_nodes) > 0:

                min_comp_dist = np.inf
                min_comp_dist_id = -1
                min_comp_dist_node = -1
                for i, l in enumerate(leaf_nodes):
                    ne = neighbors2[l]
                    ne = list(ne)[0]

                    node_pos = features[l][:3]

                    nodes_pos_diff = ((features[:, :3] - node_pos) ** 2).sum(axis=1)
                    nodes_pos_diff[ne] = np.inf
                    nodes_pos_diff[l] = np.inf
                    nodes_pos_diff[list(nodes)] = np.inf

                    min_dist_id = np.argmin(nodes_pos_diff)
                    min_dist = np.min(nodes_pos_diff)

                    if min_comp_dist > min_dist:
                        min_comp_dist = min_dist
                        min_comp_dist_id = min_dist_id
                        min_comp_dist_node = l

                if min_comp_dist_id != -1 and min_comp_dist_node != -1:
                    neighbors2[min_comp_dist_id].add(min_comp_dist_node)
                    neighbors2[min_comp_dist_node].add(min_comp_dist_id)

        adj_matrix = neighbors_to_adjacency(neighbors2, range(len(neighbors2)))
        G = nx.Graph(adj_matrix)
        num_comp = nx.number_connected_components(G)
        if verbose:
            print(count, num_comp)
        count += 1
        
    return adj_matrix, neighbors2


def rotate_cell(cell_id, morphology, df):
    """ 
    Rotate neurons vertically with respect to pia.
    Args:
        cell_id: str
        morphology: AllenSDK morphology object
        df: pandas dataframe containing angles per neuron
    """
    z_rot = df[df['specimen_id']==cell_id]['upright_angle'].values[0]
    rot1 = R.from_euler('z', z_rot, degrees=True).as_matrix()
    rot_list = list(rot1.flatten()) + [0, 0, 0]
    morphology.apply_affine(rot_list)


    x_rot = df[df['specimen_id']==cell_id]['estimated_slice_angle'].values[0]
    if not pd.isna(x_rot):
        rot2 = R.from_euler('x', x_rot, degrees=True).as_matrix()
        rot_list2 = list(rot2.flatten()) + [0, 0, 0]
        morphology.apply_affine(rot_list2)
    
    return morphology


def scale_soma_depth(cell_id, norm_features, df):
    """ 
    Move neuron to soma depth realtive to pia.
    Args:
        cell_id: str
        norm_features: normalized features per node
        df: pandas dataframe containing soma depth per neuron
    """
    # soma depth
    d_fac = df[df['specimen_id']==cell_id]['soma_distance_from_pia'].values[0]
    pos3 = norm_features[:, :3].copy()
    pos3[:, 1] -= d_fac

    norm_features[:, :3] = pos3
    
    return norm_features