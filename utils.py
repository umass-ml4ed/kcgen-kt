import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from pdb import set_trace


def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def aggregate_metrics(log):
    results = {}
    for k in log[0].keys():
        if k == 'auc':
            logits = np.concatenate([x[k]['logits'].numpy().reshape(-1) for x in log])
            scores = np.concatenate([x[k]['scores'].numpy().reshape(-1) for x in log])
            results[k] = roc_auc_score(scores, logits)

            prob = 1 / (1 + np.exp(-logits))
            pred = (prob > 0.5) * 1
            results['f1'] = f1_score(scores, pred)
            # print('F1 score:', f1)

        elif k == 'pred':
            res = np.concatenate([x[k].numpy().reshape(-1) for x in log])
            results[k] = res.sum()
        else:
            res = np.concatenate([x[k].numpy().reshape(-1) for x in log])
            results[k] = np.mean(res)
    return results
