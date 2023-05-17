import evaluate
import torch
import numpy as np
from .base import HypedMetrics
from transformers import EvalPrediction
from transformers.adapters import PredictionHead

class HypedMlcMetrics(HypedMetrics):

    def __init__(
        self,
        head:PredictionHead,
        average:str ='micro',
        k:None|int =5,
        t:None|float =None,
    ):
        super(HypedMlcMetrics, self).__init__(head)
        self.average = average
        # classification strategy hyperparameters
        self.k = min(k, head.config['num_labels'])
        self.t = t
        # check valid classification strategy
        if (self.k is None) and (self.t is None):
            raise ValueError("No classification strategy specified, got k=None and t=None.")
        if (self.k is not None) and (self.t is not None):
            raise ValueError("To many classification strategies specified, got k=%i and t=%f." % (self.k, self.t))

        # get label mapping from head config
        label2id = head.config.get('label2id', None)
        if label2id is None:
            raise ValueError("Config of head type %s has no `label2id` entry." % type(head))
        # build label space array from mapping
        self.label_space = [None] * len(label2id)
        for label, i in label2id.items():
            self.label_space[i] = label


    def compute(self, eval_pred:EvalPrediction) -> dict[str, float]:

        preds, labels = eval_pred
        preds, labels = preds.astype(bool), labels.astype(bool)
        # compute confusion matrix
        tp = (preds & labels).sum(axis=0)
        fp = (preds & (1 - labels)).sum(axis=0)
        fn = ((1 - preds) & labels).sum(axis=0)
        tn = ((1 - preds) & (1 - labels)).sum(axis=0)
        # build global confusion matrix
        total_fp = fp.sum()
        total_fn = fn.sum()
        total_tp = tp.sum()
        total_tn = tn.sum()

        # compute metrics per class
        a = (tp + tn) / preds.shape[0]
        p = tp / (tp + fp + 1e-5)
        r = tp / (tp + fn + 1e-5)
        f = 2.0 * (p * r) / (p + r + 1e-5)

        # build per-class metrics dict
        metrics = {
            label: {
                'accuracy': a[i],
                'precision': p[i],
                'recall': r[i],
                'f1': f[i],
                'confusion': {
                    'tp': tp[i],
                    'fp': fp[i],
                    'tn': tn[i],
                    'fn': fn[i]
                }
            }
            for i, label in enumerate(self.label_space)
        }
        # add global confusion matrix
        metrics['confusion'] = {
            'tp': total_tp,
            'fp': total_fp,
            'tn': total_tn,
            'fn': total_fn
        }
        # compute average accuracy
        metrics['average_accuracy'] = a.mean()

        # compute average metrics
        if self.average == 'micro':
            # micro average precision and recall
            avg_p = total_tp / (total_tp + total_fp + 1e-5)
            avg_r = total_tp / (total_tp + total_fn + 1e-5)
            # micro average
            return metrics | {
                'micro_precision':  avg_p,
                'micro_recall': avg_r,
                'micro_f1': 2.0 * (avg_p * avg_r) / (avg_p + avg_r + 1e-5)
            }

        if self.average == 'macro':
            # macro average
            return metrics | {
                'macro_precision': p.mean(),
                'macro_recall': r.mean(),
                'macro_f1': f.mean()
            }

        return metrics


    @torch.no_grad()
    def preprocess(self, logits:torch.Tensor, labels:torch.Tensor) -> np.ndarray:
        # TODO: for extreme large label spaces this is very memory expensive
        if self.t is not None:
            # theshold-based classification
            return torch.sigmoid(logits) >= self.t

        elif self.k is not None:
            # top-k based classification
            _, idx = torch.topk(logits, k=self.k, dim=-1)
            mask = torch.zeros(logits.size(), dtype=int)
            for i in range(idx.size(0)):
                mask[i, idx[i, :]] = 1
            return mask

        raise RuntimeError()


