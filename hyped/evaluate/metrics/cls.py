import evaluate
import numpy as np
from .base import HypedMetrics
from transformers import EvalPrediction
from transformers.adapters import PredictionHead

class HypedClsMetrics(HypedMetrics):

    def __init__(self, head:PredictionHead, average:str ='micro'):
        super(HypedClsMetrics, self).__init__(head)
        self.average = average
        # load all metrics
        self.a = evaluate.load("accuracy")
        self.p = evaluate.load("precision")
        self.r = evaluate.load("recall")
        self.f = evaluate.load("f1")

    def compute(self, eval_pred:EvalPrediction) -> dict[str, float]:
        # convert to nameing expected by metrics
        eval_pred = dict(
            predictions=eval_pred.predictions,
            references=eval_pred.label_ids
        )
        # compute metrics
        return {
            **self.a.compute(**eval_pred),
            **self.p.compute(**eval_pred, average=self.average),
            **self.r.compute(**eval_pred, average=self.average),
            **self.f.compute(**eval_pred, average=self.average)
        }

    def preprocess(self, logits:np.ndarray, labels:np.ndarray) -> np.ndarray:
        return logits.argmax(dim=-1)
