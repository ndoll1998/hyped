import torch.nn as nn
from wrapt import CallableObjectProxy
from abc import ABC, abstractmethod
from transformers import PreTrainedModel
from copy import copy
from .heads import HypedHeadConfig

class HypedModelWrapper(CallableObjectProxy, ABC):

    def __init__(self, model:PreTrainedModel) -> None:
        # check model type
        if not isinstance(model, PreTrainedModel):
            raise TypeError("Model of type `%s` must inherit type `%s`" % (type(model), PreTrainedModel))
        # initialize model proxy
        CallableObjectProxy.__init__(self, model)

    @property
    @abstractmethod
    def head_configs(self) -> list[HypedHeadConfig]:
        ...
