import torch
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

    
class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def subsample_graph(neighbors=None, not_deleted=None, keep_nodes=200, 
                    protected=[0]):
    """ 
    Subsample graph.
    Args:
        neighbors: dict of neighbors per node
        not_deleted: list of nodes, who did not get deleted in previous processing steps
        keep_nodes: number of nodes to keep in graph
        protected: nodes to be excluded from subsampling
    """
    if neighbors is not None:
        k_nodes = len(neighbors)
    else:
        raise ValueError('neighbors must be provided')

    # protect soma node from being removed
    protected = set(protected)

    # indices as set in random order
    perm = torch.randperm(k_nodes).tolist()
    all_indices = np.array(list(not_deleted))[perm].tolist()
    deleted = set()
    
    while len(deleted) < k_nodes - keep_nodes:

        while True:
            if len(all_indices) == 0:
                assert len(not_deleted) > keep_nodes, len(not_deleted)
                remaining = list(not_deleted - deleted)
                perm = torch.randperm(len(remaining)).tolist()
                all_indices = np.array(remaining)[perm].tolist()

            idx = all_indices.pop()

            if idx not in deleted and len(neighbors[idx]) < 3 and idx not in protected:
                break

        if len(neighbors[idx]) == 2:
            n1, n2 = neighbors[idx]
            neighbors[n1].remove(idx)
            neighbors[n2].remove(idx)
            neighbors[n1].add(n2)
            neighbors[n2].add(n1)
        elif len(neighbors[idx]) == 1:
            n1 = neighbors[idx].pop()
            neighbors[n1].remove(idx)

        del neighbors[idx]
        deleted.add(idx)

    not_deleted = list(not_deleted - deleted)
    return neighbors, not_deleted


def rotate_graph(pos_matrix, axis=None):
    """ 
    Randomly rotate graph in xyz-direction.
    Args:
        pos_matrix: matrix with xyz-node positions (N x 3).
        axis: rotate graph around axis; Possible values: x, y, z, no_rot'
    """
    rotation_matrix = R.random().as_matrix()
    
    if axis == 'x':
        rotation_matrix[0, 1] = 0
        rotation_matrix[0, 2] = 0
        rotation_matrix[0, 0] = 1
        rotation_matrix[1, 0] = 0
        rotation_matrix[2, 0] = 0
    elif axis == 'y':
        rotation_matrix[0, 1] = 0
        rotation_matrix[1, 0] = 0
        rotation_matrix[1, 1] = 1
        rotation_matrix[1, 2] = 0
        rotation_matrix[2, 1] = 0
    elif axis == 'z':
        rotation_matrix[0, 2] = 0
        rotation_matrix[1, 2] = 0
        rotation_matrix[2, 2] = 1
        rotation_matrix[2, 0] = 0
        rotation_matrix[2, 1] = 0
    elif axis == 'no_rot':
        rotation_matrix = np.eye(3)
    else:
        raise ValueError('Possible values: x, y, z, no_rot')

    rot_pos_matrix = pos_matrix @ rotation_matrix
    return rot_pos_matrix


def jitter_node_pos(pos_matrix, scale=1):
    """ 
    Randomly jitter nodes in xyz-direction.
    Args:
        pos_matrix: matrix with xyz-node positions (N x 3).
        scale: scale factor of jittering
    """
    return pos_matrix + (torch.randn(*pos_matrix.shape).numpy() * scale)


def neighbors_to_adjacency(neighbors, not_deleted):
    """ 
    Create adjacency matrix from list of non-empty neighbors. 
    Args:
        neighbors: dict of neighbors per node
        not_deleted: list of nodes, who did not get deleted in previous processing steps
    """
    node_map = {n: i for i, n in enumerate(not_deleted)}
    
    n_nodes = len(not_deleted)
    
    new_adj_matrix = np.zeros((n_nodes, n_nodes))
    for ii in neighbors.keys():
        for jj in neighbors[ii]:
            i, j = node_map[ii], node_map[jj]
            new_adj_matrix[i, i] = True  # diagonal if needed
            new_adj_matrix[i, j] = True
            new_adj_matrix[j, i] = True   
            
    return new_adj_matrix


def adjacency_to_neighbors(adj_matrix):
    """ 
    Create list of non-empty neighbors from adjacancy matrix. 
    Args:
        adj_matrix: adjacency matrix (N x N)
    """
    # remove diagonal to avoid self-neighbors
    a, b = np.where(adj_matrix - np.eye(len(adj_matrix)) == 1)
    from collections import defaultdict
    neigh = dict()
    for _a, _b in zip(a,b):
        if _a not in neigh:
            neigh[_a] = set()
        neigh[_a].add(_b)
    return neigh


def compute_eig_lapl(adj_matrix, pos_enc_dim=32):
    """ 
    Compute positional encoding using graph laplacian.
    Args:
        adj_matrix: adjacency matrix (N x N)
        pos_enc_dim: output dimensions of positional encoding.

    Adapted from https://github.com/graphdeeplearning/benchmarking-gnns/blob/ef8bd8c7d2c87948bc1bdd44099a52036e715cd0/data/molecules.py#L147-L168.
    """
    
    # Laplacian
    A = adj_matrix.astype(float)
    degree_matrix = adj_matrix.sum(axis=0).clip(1)
    N = np.diag(degree_matrix ** -0.5)
    L = np.eye(len(adj_matrix)) - (N @ adj_matrix) @ N

    # Eigenvectors with numpy
    eig_val, eig_vec = np.linalg.eig(L)
    idx = eig_val.argsort() # increasing order
    eig_val, eig_vec = eig_val[idx], np.real(eig_vec[:,idx])
    pos_enc = torch.from_numpy(eig_vec[:, 1:pos_enc_dim + 1]).float()

    return pos_enc


def get_leaf_branch_nodes(neighbors):
    """"
    Create list of candidates for leaf and branching nodes.
    Args:
        neighbors: dict of neighbors per node
    """
    all_nodes = list(neighbors.keys())
    leafs = [i for i in all_nodes if len(neighbors[i]) == 1]
    
    candidates = leafs
    next_nodes = []
    for l in leafs:
        next_nodes += [n for n in neighbors[l] if len(neighbors[n]) == 2]

    while next_nodes:
        s = next_nodes.pop(0) 
        candidates.append(s)
        next_nodes += [n for n in neighbors[s] if len(neighbors[n]) == 2 and n not in candidates and n not in next_nodes] 
        
    return candidates


def compute_node_distances(idx, neighbors):
    """"
    Computation of node degree.
    Args:
        idx: index of node
        neighbors: dict of neighbors per node
    """
    queue = []
    queue.append(idx)

    degree = dict()
    degree[idx] = 0

    while queue:
        s = queue.pop(0) 
        prev_dist = degree[s]

        for neighbor in neighbors[s]:
              if neighbor not in degree:
                queue.append(neighbor)
                degree[neighbor] = prev_dist + 1

    return degree


def drop_random_branch(nodes, neighbors, distances, keep_nodes=200):
    """ 
    Removes a branch, but ignores if branching nodes are affected. 
    solution: Starting nodes should be between branching node and leaf (see leaf_branch_nodes)

    Args:
        nodes: list of nodes of the graph
        neighbors: dict of neighbors per node
        distances: dict of distances of nodes to origin
        keep_nodes: number of nodes to keep in graph
    """
    start = list(nodes)[torch.randint(len(nodes), (1,)).item()]
    to = list(neighbors[start])[0]

    if distances[start] > distances[to]:
        start, to = to, start

    drop_nodes = [to]
    next_nodes = [n for n in neighbors[to] if n != start]

    while next_nodes:
        s = next_nodes.pop(0) 
        drop_nodes.append(s)
        next_nodes += [n for n in neighbors[s] if n not in drop_nodes] 

    if len(neighbors) - len(drop_nodes) < keep_nodes:
        return neighbors, set()
    else:
        # delete nodes
        for key in drop_nodes:
            if key in neighbors:
                for k in neighbors[key]:
                    neighbors[k].remove(key)
                del neighbors[key]

        return neighbors, set(drop_nodes)
    
    
def traverse_dir(start, to, neighbors):
    """ 
    Traverse branch start at node 'start' in direction of node 'to'. 
    Args:
        start: start node
        to: destination node
        neighbors: dict of neighbors per node
    """
    visited = [start, to]
    next_nodes = [n for n in neighbors[to] if n != start]

    while next_nodes:
        s = next_nodes.pop(0) 
        visited.append(s)
        next_nodes += [n for n in neighbors[s] if n not in visited]
    
    return visited


def cumulative_jitter(nodes_to_leaf, feats, strength=(1000, 300)):
    """ 
    Apply cumulative jitter to graph. 
    Args:
        nodes_to_leaf: indices of nodes from start node to leaf
        feats: features per node
        strength: strength scale for the jittering
    """
    jitter = (2 * torch.rand(3) - 1) * strength[0]
    for _, n in enumerate(nodes_to_leaf):
        temp = torch.tensor(feats[n])[:3] + jitter
        feats[n][:3] = tuple(temp.tolist())
        jitter += (2 * torch.rand(3) - 1) * strength[1]

    return feats


def jitter_soma_depth(feats, scale=10):
    """"
    Apply jitter to soma depth.
    Args:
        feats: features per node
        scale: scale factor of jittering
    """
    new_feats = feats.copy()
    new_feats[:, 1] += torch.randn(1).numpy() * scale
    return new_feats


def plot_neuron(neighbors, node_feats, ax1=0, ax2=1, soma_id=0, ax=None):
    """ Plot graph of 3D neuronal morphology. """   
    colors = list(sns.dark_palette('#69d', n_colors=4))
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    
    ax.set_aspect('equal')

    for i, neigh in neighbors.items():
        for j in neigh:
            n1, n2 = node_feats[i], node_feats[j]
            c = colors[np.argmax(n2[4:])]
            ax.plot([n1[ax1], n2[ax1]], [n1[ax2], n2[ax2]], color=c, linewidth=1)

    ax.scatter(node_feats[soma_id][ax1], node_feats[soma_id][ax2], color=colors[0], s=10, zorder=10)
    
    sns.despine(trim=1)
    
    
def plot_tsne(z, labels, targets, colors=None):
    """ Plot t-SNE clustering. """
    u_labels = np.unique(labels)
    fig = plt.figure(1, figsize=(8, 8))
    for label in u_labels:
        plt.scatter(z[labels == label, 0], 
                    z[labels==label, 1], 
                    s=20, 
                    label=str(targets[label]),
                    color=colors[label])
    plt.legend(bbox_to_anchor=(1,1))
    plt.axis('off')