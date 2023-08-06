import torch.nn as nn
from wrapt import CallableObjectProxy
from abc import ABC, abstractmethod
from transformers import PreTrainedModel
from .heads import HypedHeadConfig

class HypedModelWrapper(CallableObjectProxy, ABC):

    BASE_MODEL_TYPE:type[nn.Module] = PreTrainedModel

    def __init__(self, model:PreTrainedModel) -> None:
        # check model type
        if not isinstance(model, type(self).BASE_MODEL_TYPE):
            raise TypeError(
                "Model of type `%s` must inherit type `%s`" % (type(model), PreTrainedModel)
            )
        # initialize model proxy
        CallableObjectProxy.__init__(self, model)

    @property
    @abstractmethod
    def head_configs(self) -> list[HypedHeadConfig]:
        ...
