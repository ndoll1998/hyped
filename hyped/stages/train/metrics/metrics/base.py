from abc import ABC, abstractmethod
from dataclasses import dataclass
from transformers import EvalPrediction
from transformers.adapters.heads import PredictionHead
from typing import Literal, Any
from ..processors import LogitsProcessor

@dataclass
class HypedMetricConfig(object):
    metric_type:Literal['hyped-metric'] = 'hyped-metric'
    # name prefix
    prefix:None|str = None

class HypedMetric(ABC):

    def __init__(
        self,
        head:PredictionHead,
        config:HypedMetricConfig,
        processor:None|LogitsProcessor
    ) -> None:
        # save head, config and logits preprocessor
        self.head = head
        self.config = config
        self.processor = processor

    @abstractmethod
    def compute(self, eval_pred:EvalPrediction) -> dict[str, Any]:
        ...

    def add_prefix(self, key:str) -> str:
        return ("%s_%s" % (self.head.name, key)) if self.config.prefix is None else \
            ("%s_%s_%s" % (self.head.name, self.config.prefix, key))

    def __call__(self, eval_pred:EvalPrediction) -> dict[str, Any]:
        return {
            self.add_prefix(key): val
            for key, val in self.compute(eval_pred).items()
        }
