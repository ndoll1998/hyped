from .metrics.base import HypedMetric
from transformers import EvalPrediction
from collections import defaultdict

def get_labels(labels:dict, label_names:list[str]):
    if len(label_names) == 1:
        return labels[label_names[0]]
    return [labels[name] for name in label_names]

class HypedMetricCollection(object):

    def __init__(
        self,
        metrics:list[HypedMetric],
        head_order:list[str],
        label_order:list[str]
    ) -> None:
        # save head and label order
        self.head_order = head_order
        self.label_order = label_order

        self.metrics = metrics
        self.h_configs = {}
        self.processors = defaultdict(set)

        for metric in metrics:
            self.h_configs[metric.h_config.head_name] = metric.h_config
            self.processors[metric.h_config.head_name].add(metric.processor)

    def compute(self, eval_pred):
        scores = {}
        # unpack and make sure labels is list
        preds, labels = eval_pred
        labels = labels if len(self.label_order) > 1 else [labels]
        # create labels lookup
        labels = dict(zip(self.label_order, labels))
        # compute all metrics
        for metric in self.metrics:
            scores.update(metric(
                EvalPrediction(
                    predictions=preds[metric.h_config.head_name, metric.processor],
                    label_ids=get_labels(labels, metric.h_config.label_columns)
                )
            ))
        # return all scores
        return scores

    def preprocess(self, logits, labels):
        # make sure logits and labels are lists
        logits = logits if len(self.head_order) > 1 else [logits]
        labels = labels if len(self.label_order) > 1 else [labels]
        # check sizes
        assert len(logits) == len(self.head_order)
        assert len(labels) == len(self.label_order)
        # unpack logits
        logits = [l.logits if hasattr(l, 'logits') else l for l in logits]
        # create look-ups
        logits = dict(zip(self.head_order, logits))
        labels = dict(zip(self.label_order, labels))
        # preprocess all logits
        return {
            (h_name, p): p(
                logits=logits[h_name],
                labels=get_labels(labels, self.h_configs[h_name].label_columns)
            )
            for h_name, ps in self.processors.items()
            for p in ps
        }
