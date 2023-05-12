import numpy as np
from abc import ABC, abstractmethod
from transformers import EvalPrediction
from transformers.adapters import PredictionHead

class HypedMetrics(ABC):

    def __init__(self, head:PredictionHead):
        self.head = head

    def __call__(self, eval_pred:EvalPrediction) -> dict[str, float]:
        # evaluate metrics and add head prefix
        scores = self.compute(eval_pred)
        scores = {"%s_%s" % (self.head.name, key): val for key, val in scores.items()}
        # return
        return scores

    @abstractmethod
    def compute(self, eval_pred:EvalPrediction) -> dict[str, float]:
        ...

    @abstractmethod
    def preprocess(self, logits:np.ndarray, labels:np.ndarray) -> np.ndarray:
        ...
