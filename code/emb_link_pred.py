import torch
import numpy as np
from torch_geometric.nn import Node2Vec
from utils.data_utils import load_data
from utils.split_edges import load_edges
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import train_test_split_edges, negative_sampling, add_self_loops


def main():
    data_name = 'cs'
    dataset = load_data(data_name)
    data = dataset[0]
    train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index = load_edges(data_name)
    data.train_pos_edge_index = torch.from_numpy(train_pos_edge_index).t()
    data.train_neg_edge_index = torch.from_numpy(train_neg_edge_index).t()
    data.val_pos_edge_index = torch.from_numpy(val_pos_edge_index).t()
    data.val_neg_edge_index = torch.from_numpy(val_neg_edge_index).t()
    data.test_pos_edge_index = torch.from_numpy(test_pos_edge_index).t()
    data.test_neg_edge_index = torch.from_numpy(test_neg_edge_index).t()
    data.edge_index = data.train_pos_edge_index

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(data.edge_index, embedding_dim=32, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True, num_workers=4)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)
        return acc

    for epoch in range(1, 101):
        loss = train()
        # acc = test()
        # print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    emb = model().detach().cpu()
    train_edge_index = torch.cat((data.train_pos_edge_index, data.train_neg_edge_index), dim=-1)
    train_edge_label = torch.cat((
        torch.ones(data.train_pos_edge_index.size(1)),
        torch.zeros(data.train_neg_edge_index.size(1))
    ), dim=0)
    val_edge_index = torch.cat((data.val_pos_edge_index, data.val_neg_edge_index), dim=-1)
    val_edge_label = torch.cat((
        torch.ones(data.val_pos_edge_index.size(1)),
        torch.zeros(data.val_neg_edge_index.size(1))
    ), dim=0)
    test_edge_index = torch.cat((data.test_pos_edge_index, data.test_neg_edge_index), dim=-1)
    test_edge_label = torch.cat((
        torch.ones(data.test_pos_edge_index.size(1)),
        torch.zeros(data.test_neg_edge_index.size(1))
    ), dim=0)
    train_z = (emb[train_edge_index[0]] * emb[train_edge_index[1]]).sum(dim=-1).sigmoid().numpy()
    val_z = (emb[val_edge_index[0]] * emb[val_edge_index[1]]).sum(dim=-1).sigmoid().numpy()
    test_z = (emb[test_edge_index[0]] * emb[test_edge_index[1]]).sum(dim=-1).sigmoid().numpy()
    train_auc = roc_auc_score(train_edge_label, train_z)
    train_ap = average_precision_score(train_edge_label, train_z)
    val_auc = roc_auc_score(val_edge_label, val_z)
    val_ap = average_precision_score(val_edge_label, val_z)
    test_auc = roc_auc_score(test_edge_label, test_z)
    test_ap = average_precision_score(test_edge_label, test_z)
    return train_auc, train_ap, val_auc, val_ap, test_auc, test_ap


if __name__ == "__main__":
    train_aucs, train_aps, val_aucs, val_aps, test_aucs, test_aps = [], [], [], [], [], []
    for i in range(10):
        train_auc, train_ap, val_auc, val_ap, test_auc, test_ap = main()
        train_aucs.append(train_auc)
        train_aps.append(train_ap)
        val_aucs.append(val_auc)
        val_aps.append(val_ap)
        test_aucs.append(test_auc)
        test_aps.append(test_ap)

    print('Train')
    print('The mean value of auc is: {:.4f}, the std of auc is: {:.4f}'.format(np.mean(train_aucs), np.std(train_aucs)))
    print('The mean value of ap is: {:.4f}, the std of ap is: {:.4f}'.format(np.mean(train_aps), np.std(train_aps)))
    print('Valid')
    print('The mean value of auc is: {:.4f}, the std of auc is: {:.4f}'.format(np.mean(val_aucs), np.std(val_aucs)))
    print('The mean value of ap is: {:.4f}, the std of ap is: {:.4f}'.format(np.mean(val_aps), np.std(val_aps)))
    print('Test')
    print('The mean value of auc is: {:.4f}, the std of auc is: {:.4f}'.format(np.mean(test_aucs), np.std(test_aucs)))
    print('The mean value of ap is: {:.4f}, the std of ap is: {:.4f}'.format(np.mean(test_aps), np.std(test_aps)))
