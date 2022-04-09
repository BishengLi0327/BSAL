"""
Graph Convolutional Network --- non PyG Version
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gcn_normalize(adj):
    adj = adj + torch.eye(adj.shape[0], device='cuda:0')
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.diag(deg.pow(-0.5))
    adj = torch.spmm(deg_inv_sqrt, torch.spmm(adj, deg_inv_sqrt))
    return adj


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_TV(nn.Module):
    def __init__(self, nfeat, nhid, nout):
        super(GCN_TV, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)

    def forward(self, data):
        x, adj = data.x, data.adj
        adj = gcn_normalize(adj)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.gc2(x, adj)
        return x
