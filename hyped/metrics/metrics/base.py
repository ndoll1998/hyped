from abc import ABC, abstractmethod
from dataclasses import dataclass
from transformers import EvalPrediction
from typing import Any
from ..processors import LogitsProcessor
from hyped.modeling.heads import HypedHeadConfig

@dataclass
class HypedMetricConfig(object):
    # name prefix
    prefix:None|str = None

class HypedMetric(ABC):

    def __init__(
        self,
        h_config:HypedHeadConfig,
        m_config:HypedMetricConfig,
        processor:None|LogitsProcessor
    ) -> None:
        # save head, config and logits preprocessor
        self.h_config = h_config
        self.m_config = m_config
        self.processor = processor

    @abstractmethod
    def compute(self, eval_pred:EvalPrediction) -> dict[str, Any]:
        ...

    def add_prefix(self, key:str) -> str:
        return ("%s_%s" % (self.h_config.head_name, key)) if self.m_config.prefix is None else \
            ("%s_%s_%s" % (self.h_config.head_name, self.m_config.prefix, key))

    def __call__(self, eval_pred:EvalPrediction) -> dict[str, Any]:
        return {
            self.add_prefix(key): val
            for key, val in self.compute(eval_pred).items()
        }
