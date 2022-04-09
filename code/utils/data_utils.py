import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset, Dataset
from torch_geometric.datasets import Planetoid, CitationFull, Flickr, Twitch, Coauthor
from torch_geometric.utils import from_networkx, train_test_split_edges, add_self_loops, negative_sampling, k_hop_subgraph
from torch_geometric.transforms import RandomLinkSplit

import os
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import os.path as osp
from itertools import chain

from .func_utils import extract_enclosing_subgraphs, drnl_node_labeling


class Disease(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['disease.pt']

    def process(self):
        path = '../data/disease_lp/'
        edges = pd.read_csv(path + 'disease_lp.edges.csv')
        labels = np.load(path + 'disease_lp.labels.npy')
        features = sp.load_npz(path + 'disease_lp.feats.npz').todense()
        dataset = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edges.values).t().contiguous(),
            y=torch.tensor(labels)
        )
        torch.save(dataset, self.processed_paths[0])


class Airport(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['airport.pt']

    def process(self):
        data_path = '../data/airport'
        dataset_str = 'airport'
        graph = pickle.load(open(osp.join(data_path, dataset_str + '.p'), 'rb'))
        dataset = from_networkx(graph)
        dataset.x = dataset.feat
        dataset.feat = None
        torch.save(dataset, self.processed_paths[0])


def load_data(data_name):
    if data_name == 'cora':
        dataset = Planetoid('../data/Planetoid', name='Cora')
    elif data_name == 'cora_ml':
        dataset = CitationFull('../data/CitationFull', name='Cora_Ml')
    elif data_name == 'citeseer':
        dataset = CitationFull('../data/CitationFull', name='CiteSeer')
    elif data_name == 'pubmed':
        dataset = Planetoid('../data/Planetoid', name='PubMed')
    elif data_name == 'airport':
        dataset = Airport('../data/Airport')
    elif data_name == 'disease':
        dataset = Disease('../data/Disease')
    elif data_name == 'twitch_en':
        dataset = Twitch('../data/Twitch', name='EN')
    elif data_name == 'cs':
        dataset = Coauthor('../data/Coauthor', name='cs')
    else:
        raise ValueError('Invalid dataset!')
    return dataset


def sample_dataset(edge_index, ratio):
    if ratio != 1.0:
        num_edges = edge_index.size(1)
        edge_index = edge_index[:, np.random.permutation(num_edges)[: int(ratio * num_edges)]]
    return edge_index


class BSAL_Dataset(InMemoryDataset):
    def __init__(self, dataset, args, num_hops=1, split='train'):
        self.dataset = dataset
        self.data_name = str(dataset)[:-2]
        self.data = dataset[0]
        self.args = args
        self.num_hops = num_hops
        super(BSAL_Dataset, self).__init__(dataset.root)
        index = ['train', 'val', 'test'].index(split)
        self.data, self.slices = torch.load(self.processed_paths[index])

    @property
    def processed_file_names(self):
        if self.args.use_feat:
            return ['{}_train_data_use_feat.pt'.format(self.data_name),
                    '{}_val_data_use_feat.pt'.format(self.data_name),
                    '{}_test_data_use_feat.pt'.format(self.data_name)]
        else:
            return ['{}_train_data.pt'.format(self.data_name),
                    '{}_val_data.pt'.format(self.data_name),
                    '{}_test_data.pt'.format(self.data_name)]

    def process(self):
        if self.args.use_new_split:
            transform = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True, split_labels=True)
            train_data, val_data, test_data = transform(self.data)
            train_pos_edge_index = train_data.pos_edge_label_index
            train_neg_edge_index = train_data.neg_edge_label_index
            val_pos_edge_index = val_data.pos_edge_label_index
            val_neg_edge_index = val_data.neg_edge_label_index
            test_pos_edge_index = test_data.pos_edge_label_index
            test_neg_edge_index = test_data.neg_edge_label_index
            edge_index = train_pos_edge_index
        else:
            data = train_test_split_edges(self.data, val_ratio=0.05, test_ratio=0.10)
            edge_index, _ = add_self_loops(data.train_pos_edge_index)
            data.train_neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.train_pos_edge_index.size(1)
            )
            edge_index = data.train_pos_edge_index
            train_pos_edge_index = data.train_pos_edge_index
            train_neg_edge_index = data.train_neg_edge_index
            val_pos_edge_index = data.val_pos_edge_index
            val_neg_edge_index = data.val_neg_edge_index
            test_pos_edge_index = data.test_pos_edge_index
            test_neg_edge_index = data.test_neg_edge_index

        # sample edge index for dense graphs
        train_pos_edge_index = sample_dataset(train_pos_edge_index, self.args.train_percent)
        train_neg_edge_index = sample_dataset(train_neg_edge_index, self.args.train_percent)
        val_pos_edge_index = sample_dataset(val_pos_edge_index, self.args.val_percent)
        val_neg_edge_index = sample_dataset(val_neg_edge_index, self.args.val_percent)
        test_pos_edge_index = sample_dataset(test_pos_edge_index, self.args.test_percent)
        test_neg_edge_index = sample_dataset(test_neg_edge_index, self.args.test_percent)

        # extract subgraph for each link
        train_pos_list = extract_enclosing_subgraphs(self.data, self.num_hops, train_pos_edge_index, edge_index, 1, node_label='drnl')
        train_neg_list = extract_enclosing_subgraphs(self.data, self.num_hops, train_neg_edge_index, edge_index, 0, node_label='drnl')
        val_pos_list = extract_enclosing_subgraphs(self.data, self.num_hops, val_pos_edge_index, edge_index, 1, node_label='drnl')
        val_neg_list = extract_enclosing_subgraphs(self.data, self.num_hops, val_neg_edge_index, edge_index, 0, node_label='drnl')
        test_pos_list = extract_enclosing_subgraphs(self.data, self.num_hops, test_pos_edge_index, edge_index, 1, node_label='drnl')
        test_neg_list = extract_enclosing_subgraphs(self.data, self.num_hops, test_neg_edge_index, edge_index, 0, node_label='drnl')

        max_z = 0
        for data in chain(train_pos_list, train_neg_list, val_pos_list, val_neg_list, test_pos_list, test_neg_list):
            max_z = max(int(data.z.max()), max_z)
        for data in chain(train_pos_list, train_neg_list, val_pos_list, val_neg_list, test_pos_list, test_neg_list):
            if self.args.use_feat:
                data.x = torch.cat([data.x, F.one_hot(data.z, max_z + 1).to(torch.float)], dim=1)
            else:
                data.x = F.one_hot(data.z, max_z + 1).to(torch.float)
            data.z = None

        torch.save(self.collate(train_pos_list + train_neg_list), self.processed_paths[0])
        torch.save(self.collate(val_pos_list + val_neg_list), self.processed_paths[1])
        torch.save(self.collate(test_pos_list + test_neg_list), self.processed_paths[2])


class BSAL_Dynamic_Dataset(Dataset):
    def __init__(self, dataset, args, num_hops=1, split='train'):
        self.dataset = dataset
        self.data_name = str(dataset)[:-2]
        self.data = dataset[0]
        self.num_hops = num_hops
        self.args = args
        super(BSAL_Dynamic_Dataset, self).__init__(dataset.root)

        if self.args.use_new_split:
            transform = RandomLinkSplit(num_val=0.05, num_test=0.10, is_undirected=True, split_labels=True)
            train_data, val_data, test_data = transform(self.data)
            train_pos_edge_index = train_data.pos_edge_label_index
            train_neg_edge_index = train_data.neg_edge_label_index
            val_pos_edge_index = val_data.pos_edge_label_index
            val_neg_edge_index = val_data.neg_edge_label_index
            test_pos_edge_index = test_data.pos_edge_label_index
            test_neg_edge_index = test_data.neg_edge_label_index
            edge_index = train_pos_edge_index
        else:
            data = train_test_split_edges(self.data, val_ratio=0.05, test_ratio=0.1)
            edge_index, _ = add_self_loops(data.train_pos_edge_index)
            data.train_neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.train_pos_edge_index.size(1)
            )
            edge_index = data.train_pos_edge_index
            train_pos_edge_index = data.train_pos_edge_index
            train_neg_edge_index = data.train_neg_edge_index
            val_pos_edge_index = data.val_pos_edge_index
            val_neg_edge_index = data.val_neg_edge_index
            test_pos_edge_index = data.test_pos_edge_index
            test_neg_edge_index = data.test_neg_edge_index

        if split == 'train':
            pos_edge, neg_edge = sample_dataset(train_pos_edge_index, self.args.train_percent), \
                                 sample_dataset(train_neg_edge_index, self.args.train_percent)
        elif split == 'val':
            pos_edge, neg_edge = sample_dataset(val_pos_edge_index, self.args.val_percent), \
                                 sample_dataset(val_neg_edge_index, self.args.val_percent)
        else:
            pos_edge, neg_edge = sample_dataset(test_pos_edge_index, self.args.test_percent), \
                                 sample_dataset(test_neg_edge_index, self.args.test_percent)

        self.links = torch.cat((pos_edge, neg_edge), dim=1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)

    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]

        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], num_hops=self.num_hops, edge_index=self.data.train_pos_edge_index, relabel_nodes=True, num_nodes=self.data.num_nodes
        )
        src, dst = mapping.tolist()

        # remove target link from the subgraph
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]

        # calculate node labeling
        z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))

        sub_data = Data(x=self.data.x[sub_nodes], z=z, edge_index=sub_edge_index, y=y, sub_nodes=sub_nodes)
        return sub_data
