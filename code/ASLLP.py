"""
Adaptive Structure Learning for Link Prediction (ASLLP)
"""
import torch
from torch.optim import lr_scheduler
from torch.nn import BCEWithLogitsLoss

from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import os
import time
import logging
import numpy as np
from tqdm import tqdm
from config import parser
from utils.data_utils import load_data
from utils.eval_utils import evaluate_auc_ap, evaluate_hits
from utils.data_utils import BSAL_Dataset, BSAL_Dynamic_Dataset
from models.asllp import ASLLP, ASLLP2
from models.constructor import Graph_Constructor2, Graph_Constructor3
import warnings
warnings.filterwarnings('ignore')


def train(epoch, model, train_loader, semantic_graph, optimizers, device, update_steps):
    model.train()

    if epoch % update_steps != 0:
        total_loss = num_graphs = 0
        semantic_graph = semantic_graph.detach().to(device)
        optimizer = optimizers[0]
        for data in tqdm(train_loader, ncols=100, desc='Tuning model parameters'):
            data = data.to(device)
            optimizer.zero_grad()
            logits_1, logits_2, logits_3 = model(data, semantic_graph)
            loss_1 = BCEWithLogitsLoss()(logits_1.view(-1), data.y.to(torch.float))
            loss_2 = BCEWithLogitsLoss()(logits_2.view(-1), data.y.to(torch.float))
            loss_3 = BCEWithLogitsLoss()(logits_3.view(-1), data.y.to(torch.float))
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            num_graphs += data.num_graphs
    else:
        optimizer = optimizers[1]
        optimizer.zero_grad()
        total_loss = num_graphs = 0
        for data in tqdm(train_loader, ncols=100, desc='Update semantic graph structure'):
            data = data.to(device)
            logits_1, logits_2, logits_3 = model(data, semantic_graph)
            loss_1 = BCEWithLogitsLoss()(logits_1.view(-1), data.y.to(torch.float))
            loss_2 = BCEWithLogitsLoss()(logits_2.view(-1), data.y.to(torch.float))
            loss_3 = BCEWithLogitsLoss()(logits_3.view(-1), data.y.to(torch.float))
            total_loss = total_loss + loss_1 + loss_2 + loss_3
            num_graphs += data.num_graphs
        total_loss.backward()
        optimizer.step()

    return total_loss / num_graphs


@torch.no_grad()
def test(args, loader, semantic_graph, model, device):
    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(loader, ncols=100):
        data = data.to(device)
        _, _, logits = model(data, semantic_graph)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)

    if args.metric == 'hits':
        pos_pred = y_pred[y_true == 1]
        neg_pred = y_pred[y_true == 0]
        input_dict = {'y_pred_pos': pos_pred, 'y_pred_neg': neg_pred}
        results = evaluate_hits(input_dict, args.K)
        return results
    elif args.metric == 'auc_ap':
        return evaluate_auc_ap(y_pred, y_true)


def run(args):

    dataset = load_data(args.dataset)
    data = dataset[0]

    train_dataset_class = 'BSAL_Dynamic_Dataset' if args.dynamic_train else 'BSAL_Dataset'
    val_dataset_class = 'BSAL_Dynamic_Dataset' if args.dynamic_val else 'BSAL_Dataset'
    test_dataset_class = 'BSAL_Dynamic_Dataset' if args.dynamic_test else 'BSAL_Dataset'

    train_dataset = eval(train_dataset_class)(dataset, args, num_hops=1, split='train')
    val_dataset = eval(val_dataset_class)(dataset, args, num_hops=1, split='val')
    test_dataset = eval(test_dataset_class)(dataset, args, num_hops=1, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    k = int(data.num_edges / data.num_nodes) + 1
    constructor = Graph_Constructor3(dataset.num_features, 32, k)
    semantic_graph = Data(x=data.x, num_nodes=data.num_nodes)

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    model = ASLLP2(train_dataset, train_dataset[0].num_features, semantic_graph.num_features, hidden_channels=32, out_channels=32, num_layers=3).to(device)
    logger.info(model)
    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    optimizer_constructor = torch.optim.Adam(constructor.parameters(), lr=0.0001, weight_decay=args.wd)
    optimizers = [optimizer_model, optimizer_constructor]
    scheduler = lr_scheduler.StepLR(optimizer_model, step_size=5, gamma=0.8)

    if args.metric == 'auc_ap':
        best_val_auc = best_val_ap = test_auc = test_ap = 0
    elif args.metric == 'hits':
        best_val_hits = test_hits = 0
    else:
        raise ValueError('Invalid metric')
    patience = 0

    for epoch in range(1, args.epochs):
        scheduler.step()
        # semantic_graph.adj = constructor(data)
        semantic_graph.edge_index, semantic_graph.edge_weight = constructor(data)
        semantic_graph = semantic_graph.to(device)

        loss = train(epoch, model, train_loader, semantic_graph, optimizers, device, args.update_steps)
        results = test(args, val_loader, semantic_graph, model, device)

        if args.metric == 'auc_ap':
            val_auc, val_ap = results['AUC'], results['AP']
            if val_auc > best_val_auc:
                best_val_auc, best_val_ap = val_auc, val_ap
                test_results = test(args, test_loader, semantic_graph, model, device)
                test_auc, test_ap = test_results['AUC'], test_results['AP']
                patience = 0
            else:
                patience += 1
            if patience >= args.patience:
                logger.info('Early Stop! Best Val AUC: {:.4f}, Best Val AP: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.
                            format(best_val_auc, best_val_ap, test_auc, test_ap))
                break
            logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val_AUC: {val_auc:.4f}, Val_AP: {val_ap:.4f}, '
                        f'Test_AUC: {test_auc:.4f}, Test_AP: {test_ap:.4f}')
        elif args.metric == 'hits':
            val_hits = results['hits@{}'.format(args.K)]
            if val_hits > best_val_hits:
                best_val_hits = val_hits
                test_hits = test(args, test_loader, semantic_graph, model, device)['hits@{}'.format(args.K)]
                patience = 0
            else:
                patience += 1
            if patience >= args.patience:
                logger.info('Early Stop! Best Val Hits: {:.4f}, Test Hits: {:.4f}'.format(best_val_hits, test_hits))
                break
            logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val Hits: {val_hits:.4f}, Test Hits: {test_hits:.4f}')

    return [test_auc, test_ap] if args.metric == 'auc_ap' else [test_hits]


if __name__ == '__main__':
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_time = '_'.join(time.asctime().split(' '))
    log_dir = '../results/ASLLP'
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    handler = logging.FileHandler(os.path.join(log_dir, 'Log_{}_{}'.format(args.dataset.capitalize(), log_time)))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    logger.info(args)
    res = []
    for _ in range(args.runs):
        seed_everything(2)
        results = run(args)
        res.append(results)

    if args.metric == 'auc_ap':
        for i in range(len(res)):
            logger.info(f'Run: {i + 1:2d}, Test AUC: {res[i][0]:.4f}, Test AP: {res[i][1]:.4f}')
        auc, ap = 0, 0
        for j in range(len(res)):
            auc += res[j][0]
            ap += res[j][1]
        logger.info("The average AUC for test data is {:.4f}".format(auc / args.runs))
        logger.info("The average AP for test data is {:.4f}".format(ap / args.runs))
        logger.info("The std of AUC for test data is {:.4f}".format(np.std([i[0] for i in res])))
        logger.info("The std of AP for test data is {:.4f}".format(np.std([i[1] for i in res])))
    elif args.metric == 'hits':
        for i in range(len(res)):
            logger.info(f'Run: {i+1:2d}, HITS@{args.K:02d}: {res[i][0]:.4f}')
        hits = 0
        for j in range(len(res)):
            hits += res[j][0]
        logger.info("The average HITS@{:02d} for test data is {:.4f}".format(args.K, hits / args.runs))
        logger.info("The std of Hits@{:02d} for test data is {:.4f}".format(args.K, np.std([i[0] for i in res])))
