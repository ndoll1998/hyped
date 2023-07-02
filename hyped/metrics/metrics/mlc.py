import torch
from transformers import EvalPrediction
from .base import HypedMetric, HypedMetricConfig
from ..processors import TopKLogitsProcessor
from hyped.modeling.heads import HypedMlcHeadConfig
from dataclasses import dataclass, field

@dataclass
class MlcMetricConfig(HypedMetricConfig):
    metrics:list[str] = field(default_factory=lambda: [
        'accuracy',
        'precision',
        'recall',
        'f1'
    ])
    average:str = 'micro'
    k:int = 3

class MlcMetric(HypedMetric):

    def __init__(
        self,
        h_config:HypedMlcHeadConfig,
        m_config:HypedMetricConfig
    ) -> None:
        super(MlcMetric, self).__init__(
            h_config=h_config,
            m_config=m_config,
            processor=TopKLogitsProcessor(k=m_config.k)
        )

        if self.h_config.id2label is None:
            raise ValueError("`label2id` entry not specified in head `%s`." % h_config.head_name)

    def compute(self, eval_pred:EvalPrediction) -> dict[str, float]:

        preds, labels = eval_pred
        # compute confusion matrix
        tp = (preds & labels).sum(axis=0)
        fp = (preds & ~labels).sum(axis=0)
        fn = (~preds & labels).sum(axis=0)
        tn = (~preds & ~labels).sum(axis=0)
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
            for i, label in self.h_config.id2label.items()
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
        if self.m_config.average == 'micro':
            # micro average precision and recall
            avg_p = total_tp / (total_tp + total_fp + 1e-5)
            avg_r = total_tp / (total_tp + total_fn + 1e-5)
            # micro average
            return metrics | {
                'micro_precision':  avg_p,
                'micro_recall': avg_r,
                'micro_f1': 2.0 * (avg_p * avg_r) / (avg_p + avg_r + 1e-5)
            }

        if self.config.average == 'macro':
            # macro average
            return metrics | {
                'macro_precision': p.mean(),
                'macro_recall': r.mean(),
                'macro_f1': f.mean()
            }

        return metrics
