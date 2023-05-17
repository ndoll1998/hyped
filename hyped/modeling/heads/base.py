import torch
import logging
from abc import ABC, abstractmethod
from transformers.adapters.heads import PredictionHead
from dataclasses import dataclass
from datasets.features import Features
from typing import Literal, Any

logger = logging.getLogger(__name__)

@dataclass
class HypedPredictionHeadConfig(ABC):
    # specify the head type identifier
    # this identifier is resolved to the actual head type
    # by the `HypedAutoAdapterModel` (see `hyped.modeling.auto.py`)
    head_type:Literal['hyped-prediction-head'] = 'hyped-prediction-head'

    label_column:str = "labels"
    loss_coeff:float = 1.0

    num_labels:None|int = None
    id2label:None|list[str] = None

    def check_and_prepare(self, features:Features):

        # check if labels feature is present
        if self.label_column not in features:
            raise KeyError('Label column `%s` not present in features: %s' % (self.label_column, list(features.keys())))

        # get label space from labels feature
        labels_feature = features[self.label_column]
        label_space = self.get_label_space(labels_feature)

        # warn about overwriting num_labels and id2label
        if (self.num_labels is not None) and (self.num_labels != len(label_space)):
            logger.warn("Overwriting `num_labels` in %s." % type(self))
        if self.id2label is not None:
            logger.warn("Overwriting `id2label` in %s." % type(self))

        # specify label space in config
        self.num_labels = len(label_space)
        self.id2label = dict(enumerate(label_space))

    @abstractmethod
    def get_label_space(self, feature):
        ...


class HypedPredictionHead(PredictionHead, ABC):

    def __init__(
        self,
        label_column:str,
        loss_coeff:float
    ) -> None:
        if not hasattr(self, 'config'):
            raise RuntimeError("`PredictionHead` must be initialized before `HypedPredictionHead` in %s." % type(self))
        # save label column and loss coefficient in config
        self.config['label_column'] = label_column
        self.config['loss_coeff'] = loss_coeff

    @property
    def label_column(self) -> str:
        return self.config['label_column']

    @property
    def loss_coeff(self) -> float:
        return self.config['loss_coeff']

    def get_label_names(self) -> list[str]:
        return [self.label_column]

    def forward(self, *args, **kwargs):
        # prepare kwargs, rename label column as expected by base
        kwargs = kwargs.copy()
        kwargs['labels'] = kwargs.pop(self.label_column, None)
        # run base class
        out = self.wrapped_forward(*args, **kwargs)

        # apply loss coefficient
        if kwargs['labels'] is not None:
            if isinstance(out, dict):
                out['loss'] *= self.loss_coeff
            else:
                out = (out[0] * self.loss_ceoff,) + out[1:]

        # return output
        return out

    @abstractmethod
    def wrapped_forward(self, *args, **kwargs):
        ...

    def collate_labels(self, labels:list, return_tensors:str ='pt'):
        if return_tensors != 'pt':
            raise NotImplementedError()
        return torch.stack(labels, dim=0)
