import math
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Conv1d, MaxPool1d, Linear
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.nn import GCN, GAT


class DGCNN(torch.nn.Module):
    def __init__(self, train_dataset, in_channels, hidden_channels, out_channels, num_layers, GNN=GCNConv, k=0.6):
        super(DGCNN, self).__init__()

        if k < 1:  # Transform percentile to number.
            num_nodes = sorted([data.num_nodes for data in train_dataset])
            k = num_nodes[int(math.ceil(k * len(num_nodes))) - 1]
            k = max(10, k)
        self.k = int(k)

        self.convs = ModuleList()
        self.convs.append(GNN(in_channels, hidden_channels))
        for i in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0],
                            conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(conv1d_channels[0], conv1d_channels[1],
                            conv1d_kws[1], 1)
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        # self.lin1 = Linear(dense_dim, 128)
        self.lin1 = Linear(dense_dim, out_channels)
        # self.lin2 = Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        xs = [x]
        for conv in self.convs:
            xs += [torch.tanh(conv(xs[-1], edge_index))]
        x = torch.cat(xs[1:], dim=-1)

        # Global pooling.
        x = global_sort_pool(x, batch, self.k)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = self.lin1(x)
        # x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.lin2(x)
        return x


class BSAL(torch.nn.Module):
    def __init__(self, train_dataset, in_channels, hidden_channels, out_channels, num_layers):
        super(BSAL, self).__init__()
        self.dgcnn = DGCNN(train_dataset, in_channels, hidden_channels, out_channels, num_layers, GNN=GCNConv, k=0.6)
        self.gcn = GCN(in_channels=32, hidden_channels=64, out_channels=out_channels, num_layers=2)
        self.lin1 = Linear(out_channels, 1)
        self.lin2 = Linear(out_channels, 1)
        self.lin3 = Linear(out_channels, 1)
        self.att_1 = Linear(out_channels, 16)
        self.att_2 = Linear(out_channels, 16)
        self.query = Linear(16, 1)

    def forward(self, data, knn_graph):
        emb_1 = self.dgcnn(data)
        emb_2 = self.gcn(x=knn_graph.x, edge_index=knn_graph.edge_index)
        emb_2 = emb_2[data.target_nodes]

        N = emb_2.size(0)
        index_1 = torch.range(0, N-1, 2).to(torch.long)
        index_2 = torch.range(1, N, 2).to(torch.long)
        emb_2 = emb_2[index_1] + emb_2[index_2]

        att_1 = self.query(F.tanh(self.att_1(emb_1)))
        att_2 = self.query(F.tanh(self.att_2(emb_2)))
        alpha_t = torch.exp(att_1) / (torch.exp(att_1) + torch.exp(att_2))
        alpha_f = torch.exp(att_2) / (torch.exp(att_1) + torch.exp(att_2))
        x = alpha_t * emb_1 + alpha_f * emb_2

        output_1 = self.lin1(emb_1)
        output_2 = self.lin2(emb_2)
        output_3 = self.lin3(x)

        return output_1, output_2, output_3


class BSAL_2(torch.nn.Module):
    def __init__(self, train_dataset, in_channels, hidden_channels, out_channels, num_layers):
        super(BSAL_2, self).__init__()
        self.dgcnn = DGCNN(train_dataset, in_channels, hidden_channels, out_channels, num_layers, GNN=GCNConv, k=0.6)
        self.gcn = GCN(in_channels=32, hidden_channels=64, out_channels=out_channels, num_layers=2)
        self.lin1 = Linear(out_channels, 1)
        self.lin2 = Linear(out_channels, 1)
        self.lin3 = Linear(out_channels, 1)
        self.alpha = torch.nn.Parameter(data=torch.tensor(0.5), requires_grad=True)

    def forward(self, data, knn_graph):
        emb_1 = self.dgcnn(data)
        emb_2 = self.gcn(x=knn_graph.x, edge_index=knn_graph.edge_index)
        emb_2 = emb_2[data.target_nodes]

        N = emb_2.size(0)
        index_1 = torch.range(0, N-1, 2).to(torch.long)
        index_2 = torch.range(1, N, 2).to(torch.long)
        emb_2 = emb_2[index_1] + emb_2[index_2]

        x = self.alpha * emb_1 + (1 - self.alpha) * emb_2

        output_1 = self.lin1(emb_1)
        output_2 = self.lin2(emb_2)
        output_3 = self.lin3(x)

        return output_1, output_2, output_3


class BSAL_3(torch.nn.Module):
    def __init__(self, train_dataset, in_channels, hidden_channels, out_channels, num_layers):
        super(BSAL_3, self).__init__()
        self.dgcnn = DGCNN(train_dataset, in_channels, hidden_channels, out_channels, num_layers, GNN=GCNConv, k=0.6)
        self.gcn = GCN(in_channels=32, hidden_channels=64, out_channels=out_channels, num_layers=2)
        self.lin1 = Linear(out_channels, 1)
        self.lin2 = Linear(out_channels, 1)
        self.lin3 = Linear(out_channels, 1)
        self.att_1 = Linear(out_channels, 16)
        self.att_2 = Linear(out_channels, 16)
        self.query = Linear(16, 1)
        # self.alpha = torch.nn.Parameter(data=torch.tensor(0.5), requires_grad=True)

    def forward(self, data, knn_graph):
        emb_1 = self.dgcnn(data)
        emb_2 = self.gcn(x=knn_graph.x, edge_index=data.edge_index)
        emb_2 = emb_2[data.target_nodes]

        N = emb_2.size(0)
        index_1 = torch.range(0, N-1, 2).to(torch.long)
        index_2 = torch.range(1, N, 2).to(torch.long)
        emb_2 = emb_2[index_1] + emb_2[index_2]

        att_1 = self.query(F.tanh(self.att_1(emb_1)))
        att_2 = self.query(F.tanh(self.att_2(emb_2)))
        alpha_t = torch.exp(att_1) / (torch.exp(att_1) + torch.exp(att_2))
        alpha_f = torch.exp(att_2) / (torch.exp(att_1) + torch.exp(att_2))
        x = alpha_t * emb_1 + alpha_f * emb_2

        # x = self.alpha * emb_1 + (1 - self.alpha) * emb_2

        output_1 = self.lin1(emb_1)
        output_2 = self.lin2(emb_2)
        output_3 = self.lin3(x)

        return output_1, output_2, output_3


class BSAL_4(torch.nn.Module):
    def __init__(self, train_dataset, in_channels, hidden_channels, out_channels, num_layers):
        super(BSAL_4, self).__init__()
        self.dgcnn = DGCNN(train_dataset, in_channels, hidden_channels, out_channels, num_layers, GNN=GCNConv, k=0.6)
        self.gcn = GAT(in_channels=32, hidden_channels=8, out_channels=out_channels, num_layers=2)
        self.lin1 = Linear(out_channels, 1)
        self.lin2 = Linear(out_channels, 1)
        self.lin3 = Linear(out_channels, 1)
        self.att_1 = Linear(out_channels, 16)
        self.att_2 = Linear(out_channels, 16)
        self.query = Linear(16, 1)
        # self.alpha = torch.nn.Parameter(data=torch.tensor(0.5), requires_grad=True)

    def forward(self, data, knn_graph):
        emb_1 = self.dgcnn(data)
        emb_2 = self.gcn(x=knn_graph.x, edge_index=knn_graph.edge_index)
        emb_2 = emb_2[data.target_nodes]

        N = emb_2.size(0)
        index_1 = torch.range(0, N-1, 2).to(torch.long)
        index_2 = torch.range(1, N, 2).to(torch.long)
        emb_2 = emb_2[index_1] + emb_2[index_2]

        att_1 = self.query(F.tanh(self.att_1(emb_1)))
        att_2 = self.query(F.tanh(self.att_2(emb_2)))
        alpha_t = torch.exp(att_1) / (torch.exp(att_1) + torch.exp(att_2))
        alpha_f = torch.exp(att_2) / (torch.exp(att_1) + torch.exp(att_2))
        x = alpha_t * emb_1 + alpha_f * emb_2

        # x = self.alpha * emb_1 + (1 - self.alpha) * emb_2

        output_1 = self.lin1(emb_1)
        output_2 = self.lin2(emb_2)
        output_3 = self.lin3(x)

        return output_1, output_2, output_3
