import evaluate
import numpy as np
from transformers import EvalPrediction
from .base import HypedMetric, HypedMetricConfig
from ..processors import ArgMaxLogitsProcessor
from hyped.modeling.heads import HypedTaggingHeadConfig
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

@dataclass
class SeqEvalMetricConfig(HypedMetricConfig):
    # additional arguments
    suffix:bool = False
    scheme:None|Literal["IOB1","IOB2","IOE1","IOE2","IOBES","BILOU"] = None
    mode:None|str =None
    zero_division:Literal[0,1,"warn"] = 0

class SeqEvalMetric(HypedMetric):

    def __init__(self, h_config:HypedTaggingHeadConfig, m_config:SeqEvalMetricConfig) -> None:
        super(SeqEvalMetric, self).__init__(
            h_config=h_config,
            m_config=m_config,
            processor=ArgMaxLogitsProcessor()
        )
        # load seceval metric
        self.metric = evaluate.load('seqeval')

        # get label mapping from head config
        id2label = h_config.id2label
        if id2label is None:
            raise ValueError("`label2id` not set in head %s." % h_config.head_name)
        
        self.label_space = np.empty(h_config.num_labels+1, dtype=object)
        for i, l in id2label.items():
            self.label_space[i] = l

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
            suffix=self.m_config.suffix,
            scheme=self.m_config.scheme,
            mode=self.m_config.mode,
            zero_division=self.m_config.zero_division
        )
