import evaluate
from transformers import EvalPrediction
from transformers.adapters.heads import PredictionHead
from .base import HypedMetric, HypedMetricConfig
from ..processors import ArgMaxLogitsProcessor
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

@dataclass
class ClassificationMetricConfig(HypedMetricConfig):
    metric_type:Literal['cls'] = 'cls'
    metrics:list[str] = field(default_factory=lambda: [
        'accuracy',
        'precision',
        'recall',
        'f1'
    ])
    average:str = 'micro'

class ClassificationMetric(HypedMetric):

    def __init__(self, head:PredictionHead, config:ClassificationMetricConfig) -> None:
        super(ClassificationMetric, self).__init__(
            head=head,
            config=config,
            processor=ArgMaxLogitsProcessor()
        )
        # load all metrics
        self.metrics = [evaluate.load(name) for name in self.config.metrics]

    def compute(self, eval_pred:EvalPrediction) -> dict[str, float]:
        # convert to naming expected by metrics
        eval_pred = dict(
            predictions=eval_pred.predictions,
            references=eval_pred.label_ids
        )
        # evaluate all metrics
        scores = {}
        for metric in self.metrics:
            scores.update(
                metric.compute(**eval_pred) if metric.name == 'accuracy' else \
                metric.compute(**eval_pred, average=self.config.average)
            )
        # return all scores
        return scores
