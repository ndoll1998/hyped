import numpy as np
from .base import HypedMetrics
from transformers import EvalPrediction
from transformers.adapters.heads import PredictionHead

def get_labels(labels:dict, label_names:list[str]):
    if len(label_names) == 1:
        return labels[label_names[0]]
    return [labels[name] for name in label_names]

class HypedMetricsCollection(HypedMetrics):

    def __init__(
        self,
        metrics:list[HypedMetrics],
        head_order:list[str],
        label_order:list[str]
    ) -> None:
        # has no specific head
        super().__init__(None)
        # save metrics
        self.metrics = metrics
        # save label order
        self.head_order = head_order
        self.label_order = label_order
        # one metric per head
        if len(metrics) != len(self.head_order):
            raise ValueError("Expected exactly one metric per head!")

    def __call__(self,eval_pred:EvalPrediction) -> dict[str, float]:
        # avoid adding head prefix
        return self.compute(eval_pred)

    def compute(self, eval_pred:EvalPrediction) -> dict[str, float]:
        scores = {}
        # unpack and create labels lookup
        preds, labels = eval_pred
        labels = dict(zip(self.label_order, labels))
        # compute all metrics
        for metric in self.metrics:
            scores.update(metric(
                EvalPrediction(
                    predictions=preds[metric.head.name],
                    label_ids=get_labels(labels, metric.head.get_label_names())
                )
            ))
        # return all scores
        return scores

    def preprocess(self, logits:np.ndarray, labels:np.ndarray) -> np.ndarray:
        assert len(logits) == len(self.head_order)
        assert len(labels) == len(self.label_order)
        # unpack logits
        logits = [l.logits if hasattr(l, 'logits') else l for l in logits]
        # create look ups
        logits = dict(zip(self.head_order, logits))
        labels = dict(zip(self.label_order, labels))

        return {
            metric.head.name: metric.preprocess(
                logits=logits[metric.head.name],
                labels=get_labels(labels, metric.head.get_label_names())
            )
            for metric in self.metrics
        }
