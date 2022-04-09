import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_hits(input_dict, K):
    if not 'y_pred_pos' in input_dict:
        raise RuntimeError("Missing key of y_pred_pos")
    if not 'y_pred_neg' in input_dict:
        raise RuntimeError("Missing key of y_pred_neg")

    y_pred_pos, y_pred_neg = input_dict['y_pred_pos'], input_dict['y_pred_neg']

    if not (isinstance(y_pred_pos, torch.Tensor) and isinstance(y_pred_neg, torch.Tensor)):
        raise ValueError("both y_pred_pos and y_pred_neg need to be torch tensor")

    if len(y_pred_neg) < K:
        return {'hits@{}'.format(K): 1.}

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    return {'hits@{}'.format(K): hitsK}


def evaluate_auc_ap(y_pred, y_true):
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise ValueError('Both y_pred and y_true need to be torch.Tensor.')
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    result = {'AUC': auc, 'AP': ap}
    return result
