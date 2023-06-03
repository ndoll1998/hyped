import torch
from abc import ABC, abstractmethod
from typing import Any

class LogitsProcessor(ABC):

    @abstractmethod
    def preprocess(self, logits:torch.Tensor, labels:torch.Tensor) -> Any:
        ...

    @torch.no_grad()
    def __call__(self, logits:torch.Tensor, labels:torch.Tensor) -> Any:
        return self.preprocess(logits, labels)

    def __eq__(self, other):
        # must be same type and same configuration
        return (type(self) is type(other)) and \
            (vars(self) == vars(other))

    def __hash__(self):
        # build hashable state
        state = vars(self)
        state = (type(self),) + tuple((k, state[k]) for k in sorted(state.keys()))
        # return state hash
        return hash(state)

class ArgMaxLogitsProcessor(LogitsProcessor):

    def preprocess(self, logits, labels) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)

class TopKLogitsProcessor(LogitsProcessor):

    def __init__(self, k:int) -> None:
        self.k = k

    def preprocess(self, logits:torch.Tensor, labels:torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(logits.size(), dtype=bool)
        idx = torch.topk(logits, k=self.k, dim=-1).indices
        # binarize predicted indices
        for i in range(idx.size(0)):
            mask[i, idx[i, :]] = True
        return mask

class SigmoidAndThresholdLogitsProcessor(LogitsProcessor):

    def __init__(self, t:int) -> None:
        self.t = t

    def preprocess(self, logits:torch.Tensor, labels:torch.Tensor) -> Any:
        return torch.sigmoid(logits) >= self.t

