import os
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from abc import ABC, abstractmethod

from ssl_neuron.utils import subsample_graph, rotate_graph, jitter_node_pos, neighbors_to_adjacency, compute_eig_lapl, get_leaf_branch_nodes, compute_node_distances, drop_random_branch, traverse_dir, cumulative_jitter, jitter_soma_depth


DATA = 'ssl_neuron/data/'


class BaseDataset(ABC, Dataset):
    def __init__(self, config, mode='train'):

        self.config = config
        self.mode = mode
        self.n_nodes = config['data']['n_nodes']

        # augmentation parameters
        self.jitter_var = config['data']['jitter_var']
        self.axis_rot = config['data']['axis_rot']
        self.cum_jitter_strength = config['data']['cum_jitter_strength']
        self.n_drop_branch = config['data']['n_drop_branch']
        self.n_cum_jitter = config['data']['n_cum_jitter']
        self.jitter_var_soma = config['data']['jitter_var_soma']


    @abstractmethod
    def __getitem__(self, index):
        cell_id = self.cell_ids[index]
        return cell_id

    def __len__(self):
        return self.num_samples
    
    def _delete_subbranch(self, neighbors, soma_id):

        # candidates for start nodes for deletion (here only leaf-branch nodes)
        leaf_branch_nodes = get_leaf_branch_nodes(neighbors)

        # using the distances we can infer the direction of an edge
        distances = compute_node_distances(soma_id[0], neighbors)

        leaf_branch_nodes = set(leaf_branch_nodes)
        not_deleted = set(range(len(neighbors))) 
        for i in range(self.n_drop_branch):
            neighbors, drop_nodes = drop_random_branch(leaf_branch_nodes, neighbors, distances, keep_nodes=self.n_nodes)
            not_deleted -= drop_nodes
            leaf_branch_nodes -= drop_nodes

        return not_deleted, distances
    
    def _reduce_nodes(self, neighbors, soma_id):
        neighbors2 = {k: set(v) for k, v in neighbors.items()}

        not_deleted, distances = self._delete_subbranch(neighbors2, soma_id)

        # subsample graphs
        neighbors2, not_deleted = subsample_graph(neighbors=neighbors2, not_deleted=not_deleted, keep_nodes=self.n_nodes, protected=soma_id)

        # get new adjacency matrix
        adj_matrix = neighbors_to_adjacency(neighbors2, not_deleted)
        
        assert adj_matrix.shape == (self.n_nodes, self.n_nodes), '{} {}'.format(adj_matrix.shape)
        
        return neighbors2, adj_matrix, not_deleted, distances
    
    
    def _augment_node_position(self, features):
        # extract positional features (xyz-position)
        pos = features[:, :3]

        # rotate (random 3D rotation or rotation around z-axis)
        rot_pos = rotate_graph(pos, axis=self.axis_rot)

        # randomly jitter node position
        jittered_pos = jitter_node_pos(rot_pos, scale=self.jitter_var)
        
        # jitter soma depth
        jittered_pos = jitter_soma_depth(jittered_pos, scale=self.jitter_var_soma)
        
        features[:, :3] = jittered_pos

        return features
    
    
    def _cumulative_jitter(self, neighbors, not_deleted, features, distances):
        
        jitter_start_nodes = torch.randint(len(not_deleted), size=(self.n_cum_jitter,)).tolist()
        
        for k in jitter_start_nodes:
            start_node = not_deleted[k]
            neighs = neighbors[start_node]
            
            if len(neighs) > 1:

                # figure out which neighbor points to the leaf
                to = sorted([i for i in neighs], key=lambda x: distances[x])[-1]

                nodes_to_leaf = traverse_dir(start_node, to, neighbors)

                idcs = [sorted(not_deleted).index(n) for n in nodes_to_leaf]

                features = cumulative_jitter(idcs, features, strength=self.cum_jitter_strength)
            else:
                continue
                                           
        return features
    

    def _augment(self, neighbors, features, soma_id):
        
        # reduce nodes to N == n_nodes via subgraph deletion + subsampling
        neighbors2, adj_matrix, not_deleted, distances = self._reduce_nodes(neighbors, soma_id)

        # extract features of not-deleted nodes
        new_features = features[not_deleted].copy()
       
        # augment node position via roation and jittering
        new_features = self._augment_node_position(new_features)
          
        new_features = self._cumulative_jitter(neighbors2, not_deleted, new_features, distances)

        return new_features, adj_matrix
    
    
class AllenDataset(BaseDataset):
    """ Dataset for Allen data. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

        # load cell ids
        self.cell_ids = list(np.load(os.path.join(DATA, '{}_ids.npy'.format(self.mode))))
        
        # load cells
        self.cells = {}
        count = 0
        for i, cell_id in tqdm(enumerate(self.cell_ids)):

            features = np.load(os.path.join(DATA, 'skeletons', str(cell_id), 'features.npy'))
            with open(os.path.join(DATA, 'skeletons', str(cell_id), 'neighbors.pkl'), 'rb') as f:
                neighbors = pickle.load(f)

            if len(features) >= self.n_nodes:
                item = {}
                item['cell_id'] = cell_id
                item['features'] = features
                item['neighbors'] = neighbors

                self.cells[count] = item
                count += 1
                        
        self.num_samples = len(self.cells)

    
    def __getsingleitem__(self, index): 
        cell = self.cells[index]
        return cell['features'], cell['neighbors']
    
    def __getsingleitem__(self, index): 
        cell = self.cells[index]
        return cell['features'], cell['neighbors']
    
            
    def __getitem__(self, index): 
        features, neighbors = self.__getsingleitem__(index)

        # get two views
        features1, adj_matrix1 = self._augment(neighbors, features, [0])
        
        features2, adj_matrix2 = self._augment(neighbors, features, [0])
        
        # compute graph laplacian
        lapl1 = compute_eig_lapl(adj_matrix1)
        lapl2 = compute_eig_lapl(adj_matrix2)

        return (adj_matrix1, features1, lapl1), (adj_matrix2, features2, lapl2)

    
    
def build_dataloader(config, use_cuda=torch.cuda.is_available()):

    kwargs = {'num_workers':config['data']['num_workers'], 'pin_memory':True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
            AllenDataset(config, mode='train'),
            batch_size=config['data']['batch_size'], 
            shuffle=True, 
            drop_last=True)

    val_loader = torch.utils.data.DataLoader(
            AllenDataset(config, mode='val'),
            batch_size=20,
            shuffle=False,
            drop_last=True)

    return train_loader, val_loader