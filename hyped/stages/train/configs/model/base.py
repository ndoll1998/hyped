import pydantic
import datasets
import transformers
from typing import Literal
from abc import ABC, abstractmethod

class ModelConfig(pydantic.BaseModel, ABC):
    library:Literal[None]
    # base model
    pretrained_ckpt:str
    kwargs:dict ={}
    freeze:bool =False

    @pydantic.validator('pretrained_ckpt', pre=True)
    def _check_pretrained_ckpt(cls, value):
        try:
            # check if model is valid by loading config
            transformers.AutoConfig.from_pretrained(value)
        except OSError as e:
            # handle model invalid
            raise ValueError("Unkown pretrained checkpoint: %s" % value) from e

        return value

    @abstractmethod
    def build(self, info:datasets.DatasetInfo) -> transformers.PreTrainedModel:
        ...

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizer:
        return transformers.AutoTokenizer.from_pretrained(self.pretrained_ckpt, use_fast=True)

    @property
    def trainer_t(self) -> transformers.Trainer:
        return transformers.Trainer
