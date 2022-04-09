import time
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import lr_scheduler

from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from utils.data_utils import load_data, BSAL_Dataset, BSAL_Dynamic_Dataset
from utils.eval_utils import evaluate_auc_ap, evaluate_hits
from models.dgcnn import DGCNN
from models.d_dgcnn import Dynamic_DGCNN
from config import parser
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import logging


def train(model, train_loader, device, optimizer, train_dataset):
    model.train()

    total_loss = 0
    for data in tqdm(train_loader, ncols=70):
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(train_dataset)


@torch.no_grad()
def test(args, loader, model, device):
    model.eval()

    y_pred, y_true = [], []
    for data in tqdm(loader, ncols=70):
        data = data.to(device)
        logits = model(data)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(data.y.view(-1).cpu().to(torch.float))

    y_true, y_pred = torch.cat(y_true), torch.cat(y_pred)

    if args.metric == 'hits':
        pos_pred = y_pred[y_true == 1]
        neg_pred = y_pred[y_true == 0]
        input_dict = {'y_pred_pos': pos_pred, 'y_pred_neg': neg_pred}
        results = evaluate_hits(input_dict, 20)
        return results
    elif args.metric == 'auc_ap':
        return evaluate_auc_ap(y_pred, y_true)


def run(args):

    dataset = load_data(args.dataset)

    train_dataset_class = 'BSAL_Dynamic_Dataset' if args.dynamic_train else 'BSAL_Dataset'
    val_dataset_class = 'BSAL_Dynamic_Dataset' if args.dynamic_val else 'BSAL_Dataset'
    test_dataset_class = 'BSAL_Dynamic_Dataset' if args.dynamic_test else 'BSAL_Dataset'

    train_dataset = eval(train_dataset_class)(dataset, args, num_hops=1, split='train')
    val_dataset = eval(val_dataset_class)(dataset, args, num_hops=1, split='val')
    test_dataset = eval(test_dataset_class)(dataset, args, num_hops=1, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

    device = torch.device('cuda:0' if args.cuda else 'cpu')
    if args.dynamic_train:
        model = Dynamic_DGCNN(hidden_channels=32, num_layers=3, max_z=1000, k=0.6, train_dataset=train_dataset, use_feature=args.use_feat).to(device)
    else:
        model = DGCNN(train_dataset, train_dataset[0].num_features, hidden_channels=32, num_layers=3).to(device)

    logger.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    schedular = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    if args.metric == 'auc_ap':
        best_val_auc = best_val_ap = test_auc = test_ap = 0
    elif args.metric == 'hits':
        best_val_hits = test_hits = 0
    else:
        raise ValueError('Invalid metric')
    patience = 0

    for epoch in range(1, args.epochs):
        schedular.step()
        loss = train(model, train_loader, device, optimizer, train_dataset)
        results = test(args, val_loader, model, device)

        if args.metric == 'auc_ap':
            val_auc, val_ap = results['AUC'], results['AP']
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_ap = val_ap
                test_results = test(args, test_loader, model, device)
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
                test_hits = test(args, test_loader, model, device)['hits@{}'.format(args.K)]
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

    exp_time = '_'.join(time.asctime().split(' '))
    handler = logging.FileHandler('../results/SEAL/Log_SEAL_{}_Use_feat_{}_{}.txt'.format(args.dataset.capitalize(), str(args.use_feat), exp_time))
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
