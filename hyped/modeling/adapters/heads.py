from __future__ import annotations
from abc import ABC, abstractmethod
# hyped head configs
from ..heads import (
    HypedHeadConfig,
    HypedClsHeadConfig,
    HypedMlcHeadConfig,
    HypedTaggingHeadConfig,
    HypedCausalLMHeadConfig
)
# adapter transformers heads
from transformers import PreTrainedModel
from transformers.adapters.heads import (
    PredictionHead,
    ClassificationHead,
    MultiLabelClassificationHead,
    TaggingHead,
    CausalLMHead
)

from typing import Any
from dataclasses import dataclass, asdict

class CreateConfigFromHeadMixin(object):
    @classmethod
    def from_head(cls, head:PredictionHead, **kwargs) -> HypedHeadConfig:
        # copy head configuration and process it
        config = head.config.copy()
        config.pop("head_type")
        # convert label2id to is2label
        config['id2label'] = (
            {i: l for l,i in config.pop('label2id').items()}
            if config.get('label2id', None) is not None else
            None
        )
        # overwrite/add entries from keyword arguments
        config.update(kwargs)
        config['head_name'] = config.get('head_name', head.name)
        # create config instance
        return cls(**config)

@dataclass
class HypedAdapterClsHeadConfig(HypedClsHeadConfig, CreateConfigFromHeadMixin):
    layers:int = 2
    activation_function:str = "tanh"
    use_pooler:bool = False
    bias:bool = True

@dataclass
class HypedAdapterMlcHeadConfig(HypedMlcHeadConfig, CreateConfigFromHeadMixin):
    layers:int = 2
    activation_function:str = "tanh"
    use_pooler:bool = False
    bias:bool = True

@dataclass
class HypedAdapterTaggingHeadConfig(HypedTaggingHeadConfig, CreateConfigFromHeadMixin):
    layers:int = 2
    activation_function:str = "tanh"

@dataclass
class HypedAdapterCausalLMHeadConfig(HypedCausalLMHeadConfig, CreateConfigFromHeadMixin):
    layers:int = 1
    activation_function:str = "tanh"
    layer_norm:bool = False
    bias:bool = True
    shift_labels:bool = True

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


class HypedAdapterMlcHead(HypedAdapterHead, MultiLabelClassificationHead):

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
        MultiLabelClassificationHead.__init__(self, model, **kwargs)
        HypedAdapterHead.__init__(self, h_config)

    def get_labels(self, kwargs:dict[str, Any]) -> dict[str, Any]:
        return {'labels': kwargs.get(self.h_config.label_column)}


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


class HypedAdapterCausalLMHead(HypedAdapterHead, CausalLMHead):

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
        CausalLMHead.__init__(self, model, **kwargs)
        HypedAdapterHead.__init__(self, h_config)

        if (not self.h_config.shift_labels) and (self.h_config.label_column == "input_ids"):
            warnings.warn("Causal LM head got label_column='input_ids' and shift_labels=False. This specifies the trivial task of reproducing the input, NOT next word prediction.", UserWarning)

    def get_labels(self, kwargs:dict[str, Any]) -> dict[str, Any]:
        return {'labels': kwargs.get(self.label_column)}

