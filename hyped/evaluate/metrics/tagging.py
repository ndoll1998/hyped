import evaluate
import numpy as np
from .base import HypedMetrics
from transformers import EvalPrediction
from transformers.adapters import PredictionHead

class HypedTaggingMetrics(HypedMetrics):

    def __init__(self, head:PredictionHead):
        super(HypedTaggingMetrics, self).__init__(head)
        # load all metrics
        self.seqeval = evaluate.load("seqeval")

        # get label mapping from head config
        label2id = head.config.get('label2id', None)
        if label2id is None:
            raise ValueError("Config of head type %s has no `label2id` entry." % type(head))
        # build label space array from mapping
        self.label_space = np.empty(len(label2id), dtype=str)
        for label, i in label2id.items():
            self.label_space[i] = label

    def compute(self, eval_pred:EvalPrediction) -> dict[str, float]:
        # unpack predicitons and labels
        preds, labels = eval_pred
        # compute valid mask and lengths
        mask = (labels >= 0)
        lengths = mask.sum(axis=-1)
        # compute metric
        return self.seqeval.compute(
            # apply valid mask, convert label ids to label names
            # and split into seperate examples (masking flattens the arrays)
            predictions=np.array_split(self.label_space[preds[mask]], lengths[:-1]),
            references=np.array_split(self.label_space[labels[mask]], lengths[:-1])
        )

    def preprocess(self, logits:np.ndarray, labels:np.ndarray) -> np.ndarray:
        return logits.argmax(dim=-1)
