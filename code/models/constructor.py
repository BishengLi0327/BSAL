import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse


class Graph_Constructor(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(in_channels, out_channels)
        self.alpha = nn.Parameter(torch.tensor([0.5]))
        self.K = K

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x = data.x
        M1 = F.tanh(self.lin1(x))
        M2 = F.tanh(self.lin2(x))
        A = F.relu(F.tanh(torch.mm(M1, torch.transpose(M2, 0, 1)) - torch.mm(M2, torch.transpose(M1, 0, 1))))
        adj = A.detach()
        mask = torch.zeros(data.num_nodes, data.num_nodes)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.K, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        mask = mask - torch.diag_embed(torch.diag(mask))  # 去除掉对角线上的元素
        A = (A * mask)
        mean = torch.sum(A.detach()) / (self.K * A.shape[0])
        A = A / mean
        return A

    def fullA(self, data):
        x = data.x
        M1 = F.tanh(self.lin1(x))
        M2 = F.tanh(self.lin2(x))
        A = F.relu(F.tanh(torch.mm(M1, torch.transpose(M2, 0, 1)) - torch.mm(M2, torch.transpose(M1, 0, 1))))
        return A


class Graph_Constructor2(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.lin = nn.Parameter(torch.eye(in_channels, in_channels), requires_grad=True)
        self.K = K

    def forward(self, data):
        x = data.x
        x = torch.mm(x, self.lin)
        A = F.relu(F.tanh(torch.mm(x, torch.transpose(x, 0, 1))))
        adj = A.detach()
        # 取每一行最大的K个元素，有可能取到自身的边
        mask = torch.zeros(data.num_nodes, data.num_nodes)
        s1, t1 = adj.topk(self.K+1, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        mask = mask - torch.diag_embed(torch.diag(mask))  # 去除掉对角线上的元素
        A = (A * mask)
        mean = torch.sum(A.detach()) / (self.K * A.shape[0])
        A = A / mean
        return A

    def fullA(self, data):
        x = data.x
        x = torch.mm(self.lin, x)
        A = F.relu(torch.tanh(torch.spmm(x, torch.transpose(x, 0, 1))))
        return A


class Graph_Constructor3(torch.nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.lin = nn.Parameter(torch.eye(in_channels, in_channels), requires_grad=True)
        self.K = K

    def forward(self, data):
        x = data.x
        x = torch.mm(x, self.lin)
        A = F.relu(F.tanh(torch.mm(x, torch.transpose(x, 0, 1))))
        adj = A.detach()
        # 取每一行最大的K个元素，有可能取到自身的边
        mask = torch.zeros(data.num_nodes, data.num_nodes)
        s1, t1 = adj.topk(self.K+1, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        mask = mask - torch.diag_embed(torch.diag(mask))  # 去除掉对角线上的元素
        A = (A * mask)
        mean = torch.sum(A.detach()) / (self.K * A.shape[0])
        A = A / mean
        edge_index, edge_weight = dense_to_sparse(A)
        return edge_index, edge_weight

    def fullA(self, data):
        x = data.x
        x = torch.mm(self.lin, x)
        A = F.relu(torch.tanh(torch.spmm(x, torch.transpose(x, 0, 1))))
        return A