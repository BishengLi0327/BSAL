from .data_utils import BSAL_Dataset
import os
import os.path as osp
import numpy as np


def get_split_edges(dataset, args):
    train_dataset = BSAL_Dataset(dataset, args, num_hops=1, split='train')
    val_dataset = BSAL_Dataset(dataset, args, num_hops=1, split='val')
    test_dataset = BSAL_Dataset(dataset, args, num_hops=1, split='test')

    train_edges = []
    for i in range(len(train_dataset)):
        edge = train_dataset[i].target_nodes.tolist()
        train_edges.append(edge)
    num_edges = len(train_edges)
    train_pos_edge_index = train_edges[: int(num_edges/2)]
    train_neg_edge_index = train_edges[int(num_edges/2):]

    val_edges = []
    for i in range(len(val_dataset)):
        edge = val_dataset[i].target_nodes.tolist()
        val_edges.append(edge)
    num_edges = len(val_edges)
    val_pos_edge_index = val_edges[: int(num_edges/2)]
    val_neg_edge_index = val_edges[int(num_edges/2):]

    test_edges = []
    for i in range(len(test_dataset)):
        edge = test_dataset[i].target_nodes.tolist()
        test_edges.append(edge)
    num_edges = len(test_edges)
    test_pos_edge_index = test_edges[: int(num_edges/2)]
    test_neg_edge_index = test_edges[int(num_edges/2):]

    save_dir = osp.join('/root/libisheng/BSAL/data/Split_Edges', dataset)
    if not osp.isdir(save_dir):
        os.mkdir(save_dir)
    np.save(osp.join(save_dir, 'train_pos_edges.npy'), train_pos_edge_index)
    np.save(osp.join(save_dir, 'train_neg_edges.npy'), train_neg_edge_index)
    np.save(osp.join(save_dir, 'val_pos_edges.npy'), val_pos_edge_index)
    np.save(osp.join(save_dir, 'val_neg_edges.npy'), val_neg_edge_index)
    np.save(osp.join(save_dir, 'test_pos_edges.npy'), test_pos_edge_index)
    np.save(osp.join(save_dir, 'test_neg_edges.npy'), test_neg_edge_index)


def load_edges(dataset):
    save_dir = osp.join('/root/libisheng/BSAL/data/Split_Edges', dataset)
    train_pos_edge_index = np.load(osp.join(save_dir, 'train_pos_edges.npy'))
    train_neg_edge_index = np.load(osp.join(save_dir, 'train_neg_edges.npy'))
    val_pos_edge_index = np.load(osp.join(save_dir, 'val_pos_edges.npy'))
    val_neg_edge_index = np.load(osp.join(save_dir, 'val_neg_edges.npy'))
    test_pos_edge_index = np.load(osp.join(save_dir, 'test_pos_edges.npy'))
    test_neg_edge_index = np.load(osp.join(save_dir, 'test_neg_edges.npy'))
    return train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index
