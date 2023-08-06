import evaluate
from transformers import EvalPrediction
from .base import HypedMetric, HypedMetricConfig
from ..processors import ArgMaxLogitsProcessor
from hyped.modeling.heads import HypedClsHeadConfig
from dataclasses import dataclass, field
from functools import partial

@dataclass
class ClsMetricConfig(HypedMetricConfig):
    metrics:list[str] = field(default_factory=lambda: [
        'accuracy',
        'precision',
        'recall',
        'f1'
    ])
    average:str = 'micro'

class ClsMetric(HypedMetric):

    def __init__(
        self,
        h_config:HypedClsHeadConfig,
        m_config:ClsMetricConfig
    ) -> None:
        super(ClsMetric, self).__init__(
            h_config=h_config,
            m_config=m_config,
            processor=ArgMaxLogitsProcessor()
        )
        # load all metrics
        self.metrics = [evaluate.load(name) for name in self.m_config.metrics]

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
                metric.compute(**eval_pred, average=self.m_config.average)
            )
        # return all scores
        return scores
