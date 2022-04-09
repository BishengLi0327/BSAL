import torch

from utils.split_edges import load_edges
from utils.data_utils import load_data

import argparse

import numpy as np
from torch_geometric import seed_everything
from torch_geometric.nn import GAE
from models.encoders import *
from torch_geometric.utils import train_test_split_edges, add_self_loops, negative_sampling
from torch_geometric.data import InMemoryDataset


class HUAWEI_Dataset(InMemoryDataset):
    def __init__(self):
        super().__init__()
        self.data = torch.load('../data/HUAWEI/huawei_graph.pt')


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    loss = model.recon_loss(z, data.train_pos_edge_index, data.train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return float(loss), z


def test(metric, model, x, edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, edge_index)
        pos_pred = model.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = model.decoder(z, neg_edge_index, sigmoid=True)

        # results = {}
        # input_dict = {'y_pred_pos': pos_pred, 'y_pred_neg': neg_pred}
        # for K in [20, 50, 100]:
        #     # evaluator.K = K
        #     # hits = evaluator.eval(input_dict)[f'hits@{K}']
        #     hitsK = hits(input_dict, K)[f'hits@{K}']
        #     results[f'Hits@{K}'] = hitsK

    return model.test(z, pos_edge_index, neg_edge_index)
    # return results


def run():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--variational', action='store_true')
    # parser.add_argument('--linear', action='store_true')
    parser.add_argument('--dataset', default='disease', type=str)
    parser.add_argument('--embedding', default='node2vec', type=str, choices=['one-hot', 'count', 'random', 'node2vec'])
    parser.add_argument('--encoder', default='GAT', type=str, choices=['GCN', 'SGC', 'GAT'])
    parser.add_argument('--epochs', default=4001, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool)
    parser.add_argument('--val_ratio', default=0.05, type=float)
    parser.add_argument('--test_ratio', default=0.1, type=float)
    parser.add_argument('--metric', default='auc', choices=['auc', 'hits'])
    parser.add_argument('--patience', default=400, type=int)
    args = parser.parse_args()
    print(args)

    # dataset = load_data(args.dataset)
    # data = dataset[0]
    # train_pos_edge_index, train_neg_edge_index, val_pos_edge_index, val_neg_edge_index, test_pos_edge_index, test_neg_edge_index = load_edges(args.dataset)
    # data.train_pos_edge_index = torch.from_numpy(train_pos_edge_index).t()
    # data.train_neg_edge_index = torch.from_numpy(train_neg_edge_index).t()
    # data.val_pos_edge_index = torch.from_numpy(val_pos_edge_index).t()
    # data.val_neg_edge_index = torch.from_numpy(val_neg_edge_index).t()
    # data.test_pos_edge_index = torch.from_numpy(test_pos_edge_index).t()
    # data.test_neg_edge_index = torch.from_numpy(test_neg_edge_index).t()
    # data.edge_index = data.train_pos_edge_index
    dataset = HUAWEI_Dataset()
    data = dataset[0]
    data.x = torch.randn([data.num_nodes, 240])
    data = train_test_split_edges(data, 0.05, 0.10)
    edge_index, _ = add_self_loops(data.train_pos_edge_index)
    data.train_neg_edge_index = negative_sampling(
        edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1))
    data.edge_index = data.train_pos_edge_index

    if args.encoder == 'GCN':
        model = GAE(GCN(dataset.num_features, 32))
    elif args.encoder == 'SGC':
        model = GAE(SGC(dataset.num_features, 32))
    elif args.encoder == 'GAT':
        model = GAE(GAT(dataset.num_features, 32))
    else:
        raise ValueError('Invalid model type!')

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auc = test_auc = test_ap = 0

    best_val_hits = 0
    test_hits20, test_hits50, test_hits100 = 0, 0, 0
    patience = 0

    for epoch in range(1, args.epochs):
        train_loss, _ = train(model, data, optimizer)
        if args.metric == 'auc':
            val_auc, val_ap = test(args.metric, model, data.x, data.train_pos_edge_index,
                                   data.val_pos_edge_index, data.val_neg_edge_index)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                train_auc, train_ap = test(args.metric, model, data.x, data.train_pos_edge_index,
                                           data.train_pos_edge_index, data.train_neg_edge_index)
                test_auc, test_ap = test(args.metric, model, data.x, data.train_pos_edge_index,
                                         data.test_pos_edge_index, data.test_neg_edge_index)
                patience = 0
            else:
                patience += 1
            if patience > args.patience:
                print('early stop, best test auc: {:.4f}, best ap: {:.4f}'.format(test_auc, test_ap))
                break

            print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Val_AUC: {val_auc:.4f}, '
                  f'Val_AP: {val_ap:.4f}, Test_AUC: {test_auc:.4f}, Test_AP: {test_ap:.4f}')

    return train_auc, train_ap, val_auc, val_ap, test_auc, test_ap if args.metric == 'auc' else [test_hits20, test_hits50, test_hits100]


if __name__ == '__main__':
    train_aucs, train_aps, val_aucs, val_aps, test_aucs, test_aps = [], [], [], [], [], []
    for i in range(10):
        seed_everything(2)
        train_auc, train_ap, val_auc, val_ap, test_auc, test_ap = run()
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
