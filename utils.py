from tqdm import tqdm
import collections
import pandas as pd
import os
import torch
import random
import numpy as np
from sklearn.metrics import f1_score


def set_allseed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(pred, num_labels):
    predict = pred.predictions.argmax(axis=1)
    ref = pred.label_ids
    pred_li, ref_li = [], []
    for i, j in zip (predict, ref):
        prediction, reference = [0] * num_labels, [0] * num_labels
        prediction[i] = 1
        reference[j] = 1
        pred_li.append(prediction)
        ref_li.append(reference)
    f1 = f1_score(pred_li, ref_li, average="weighted")
    return {'f1' : f1 }