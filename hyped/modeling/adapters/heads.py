from abc import ABC, abstractmethod
# hyped head configs
from ..heads import (
    HypedHeadConfig,
    HypedClsHeadConfig,
    HypedTaggingHeadConfig
)
# adapter transformers heads
from transformers import PreTrainedModel
from transformers.adapters.heads import (
    PredictionHead,
    ClassificationHead,
    TaggingHead
)

from typing import Any
from dataclasses import dataclass, asdict

class ForwardWrapper(object):

    def __init__(self, forward_fn, get_labels_fn):
        self.forward = forward_fn
        self.get_labels = get_labels_fn

    def __call__(self, *args, **kwargs):
        # prepare kwargs, rename label column as expected by base
        kwargs = kwargs.copy()
        kwargs.update(self.get_labels(kwargs))
        # run forward with renamed labels
        return self.forward(*args, **kwargs)

class HypedAdapterHead(PredictionHead):

    def __init__(self, h_config:HypedHeadConfig) -> None:
        self.h_config = h_config
        # wrap forward function
        self.forward = ForwardWrapper(
            forward_fn=self.forward,
            get_labels_fn=self.get_labels
        )

    @abstractmethod
    def get_labels(self, kwargs:dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class HypedAdapterClsHeadConfig(HypedClsHeadConfig):
    layers:int = 2
    activation_function:str ="tanh"
    use_pooler:bool =False
    bias:bool =True

class HypedAdapterClsHead(HypedAdapterHead, ClassificationHead):

    def __init__(
        self,
        model:PreTrainedModel,
        h_config:HypedAdapterClsHeadConfig
    ) -> None:
        # extract head kwargs from config
        kwargs = asdict(h_config)
        kwargs.pop("loss_coeff")
        kwargs.pop("label_column")
        kwargs.pop("head_type", None)
        # initialize head
        ClassificationHead.__init__(self, model, **kwargs)
        HypedAdapterHead.__init__(self, h_config)

    def get_labels(self, kwargs:dict[str, Any]) -> dict[str, Any]:
        return {'labels': kwargs.get(self.h_config.label_column)}

@dataclass
class HypedAdapterTaggingHeadConfig(HypedClsHeadConfig):
    layers:int = 2
    activation_function:str ="tanh"

class HypedAdapterTaggingHead(HypedAdapterHead, TaggingHead):

    def __init__(
        self,
        model:PreTrainedModel,
        h_config:HypedAdapterTaggingHeadConfig
    ) -> None:
        # extract head kwargs from config
        kwargs = asdict(h_config)
        kwargs.pop("loss_coeff")
        kwargs.pop("label_column")
        kwargs.pop("head_type", None)
        # initialize head
        TaggingHead.__init__(self, model, **kwargs)
        HypedAdapterHead.__init__(self, h_config)

    def get_labels(self, kwargs:dict[str, Any]) -> dict[str, Any]:
        return {'labels': kwargs.get(self.label_column)}

