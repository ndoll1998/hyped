import evaluate
import numpy as np
from transformers import EvalPrediction
from transformers.adapters.heads import PredictionHead
from .base import HypedMetric, HypedMetricConfig
from ..processors import ArgMaxLogitsProcessor
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

@dataclass
class SeqEvalMetricConfig(HypedMetricConfig):
    metric_type:Literal['seqeval'] = 'seqeval'
    # additional arguments
    suffix:bool = False
    scheme:None|Literal["IOB1","IOB2","IOE1","IOE2","IOBES","BILOU"] = None
    mode:None|str =None
    zero_division:Literal[0,1,"warn"] = 0

class SeqEvalMetric(HypedMetric):

    def __init__(self, head:PredictionHead, config:SeqEvalMetricConfig) -> None:
        super(SeqEvalMetric, self).__init__(
            head=head,
            config=config,
            processor=ArgMaxLogitsProcessor()
        )
        # load seceval metric
        self.metric = evaluate.load('seqeval')

        # get label mapping from head config
        label2id = head.config.get('label2id', None)
        if label2id is None:
            raise ValueError("Config of head type %s has no `label2id` entry." % type(head))
        # build label space array from mapping
        self.label_space = np.empty(len(label2id), dtype=object)
        for label, i in label2id.items():
            self.label_space[i] = label

    def compute(self, eval_pred:EvalPrediction) -> dict[str, float]:
        # unpack predicitons and labels
        preds, labels = eval_pred
        # compute valid mask and lengths
        mask = (labels >= 0)
        splits = np.cumsum(mask.sum(axis=-1)[:-1])
        # compute metric
        return self.metric.compute(
            # apply valid mask, convert label ids to label names
            # and split into seperate examples (masking flattens the arrays)
            predictions=np.array_split(self.label_space[preds[mask]], splits),
            references=np.array_split(self.label_space[labels[mask]], splits),
            # additional arguments
            suffix=self.config.suffix,
            scheme=self.config.scheme,
            mode=self.config.mode,
            zero_division=self.config.zero_division
        )
