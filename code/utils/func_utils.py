import torch
import numpy as np
from tqdm import tqdm
from scipy.sparse.csgraph import shortest_path

from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, k_hop_subgraph, to_dense_adj, degree


def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    # Double-radius node labeling (DRNL)
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst-1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    # max_z = max(int(z.max()), max_z)
    return z.to(torch.long)


def de_node_labeling(edge_index, src, dst, num_nodes=None, max_dist=3):
    # Distance Encoding. See "Li et. al., Distance Encoding: Design Provably More
    # Powerful Neural Networks for Graph Representation Learning."
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()

    dist = shortest_path(adj, directed=False, unweighted=True, indices=[src, dst])
    dist = torch.from_numpy(dist)

    dist[dist > max_dist] = max_dist
    dist[torch.isnan(dist)] = max_dist + 1

    return dist.to(torch.long).t()


def extract_enclosing_subgraphs(data, num_hops, link_index, edge_index, y, node_label):
    data_list = []
    for src, dst in tqdm(link_index.t().tolist(), desc='Extracting...'):
        # src: source node   dst: destination node
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], num_hops=num_hops, edge_index=edge_index, relabel_nodes=True, num_nodes=data.num_nodes
        )
        target_nodes = torch.tensor([src, dst])
        src, dst = mapping.tolist()

        # remove target link from the subgraph
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]

        # calculate node labeling
        if node_label == 'drnl':
            z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))
        elif node_label == 'de':
            z = de_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))
        else:
            z = torch.zeros(sub_nodes.size(0), dtype=torch.long)

        sub_data = Data(x=data.x[sub_nodes], z=z, edge_index=sub_edge_index, y=y, sub_nodes=sub_nodes, target_nodes=target_nodes)
        data_list.append(sub_data)

    return data_list


def pagerank(data):
    N = data.num_nodes
    adj = to_dense_adj(data.edge_index, max_num_nodes=N).squeeze(dim=0)
    max_iteration = 1000
    d = degree(data.edge_index[0], num_nodes=N)
    A = torch.spmm(adj, torch.inverse(torch.diag(d)))
    prop_pre = (torch.ones(N) / N).unsqueeze(dim=1)
    eps = 0.00001
    iteration = 0
    while iteration <= max_iteration:
        prop_next = torch.mm(A, prop_pre)
        if torch.norm(prop_next-prop_pre, p=None) < eps:
            break
        prop_pre = prop_next
        iteration += 1
    return prop_pre  # dim: [data.num_nodes, 1]
