import torch
import numpy as np
import pyarrow as pa
import datasets
from transformers.adapters.heads import (
    PredictionHead,
    TaggingHead
)
from abc import ABC, abstractmethod
from typing import Any

class LogitsProcessor(ABC):

    def __init__(self, head:PredictionHead) -> None:
        # save head and get label id mapping
        self.head = head

    @property
    def label_space(self) -> np.ndarray:

        # get label mapping from head config
        label2id = self.head.config.get('label2id', None)

        # check if label mapping is set
        if label2id is None:
            raise ValueError("`label2id` not set in config of head `%s`." % self.head.head_name)

        # build label space array from mapping
        label_space = np.empty(len(label2id), dtype=object)
        for label, i in label2id.items():
            label_space[i] = label

        # return label space
        return label_space


    @abstractmethod
    def preprocess(self, logits:torch.Tensor, labels:torch.Tensor) -> Any:
        ...

    @abstractmethod
    def convert_to_dataset(self, ids:np.ndarray) -> datasets.Dataset:
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
        # get predictions by argmax
        preds = torch.argmax(logits, dim=-1)
        assert preds.shape == labels.shape
        # apply labels mask and return
        preds[labels < 0] = -100
        return preds

    @property
    def identifier(self) -> str:
        return "%s-argmax" % self.head.name

    @property
    def features(self) -> datasets.Features:

        if isinstance(self.head, TaggingHead):
            return datasets.Features({
                self.identifier: datasets.Sequence(
                    datasets.ClassLabel(
                        names=self.label_space.tolist()
                    ),
                    length=-1 # TODO
                )
            })

        return datasets.Features({
            self.identifier: datasets.ClassLabel(
                names=self.label_space.tolist()
            )
        })

    def convert_to_dataset(self, ids:np.ndarray) -> datasets.Dataset:
        # create pyarrow table with correct schema
        table = pa.table(
            data={self.identifier: ids.tolist()},
            schema=self.features.arrow_schema
        )
        # wrap in dataset
        return datasets.Dataset(table)

class TopKLogitsProcessor(LogitsProcessor):

    def __init__(self, head:PredictionHead, k:int) -> None:
        super(TopKLogitsProcessor, self).__init__(head)
        self.k = k

    def preprocess(self, logits:torch.Tensor, labels:torch.Tensor) -> torch.Tensor:
        # TODO: rework to apply to extreme multi label scenario
        mask = -torch.ones_like(logits)
        vals, idx = torch.topk(logits, k=self.k, dim=-1)
        # TODO
        vals = torch.sigmoid(vals)
        assert (vals >= 0).all()
        # binarize predicted indices
        for i in range(idx.size(0)):
            mask[i, idx[i, :]] = vals[i, :]
        return mask

    @property
    def identifier(self) -> str:
        return "%s-k=%i" % (self.head.name, self.k)

    @property
    def features(self) -> datasets.Features:
        return datasets.Features({
            self.identifier: datasets.Features({
                'labels': datasets.Sequence(
                    datasets.ClassLabel(
                        names=self.label_space.tolist()
                    ),
                    length=self.k if self.k is not None else -1
                ),
                'scores': datasets.Sequence(
                    datasets.Value(dtype='float'),
                    length=self.k if self.k is not None else -1
                )
            })
        })

    def convert_to_dataset(self, masked_logits:np.ndarray) -> datasets.Dataset:

        data = []

        mask = (masked_logits >= 0)
        for j in range(mask.shape[0]):
            ids, = mask[j, :].nonzero()
            data.append({
                'labels': ids,
                'scores': masked_logits[j, ids]
            })

        # create dataset from dict
        return datasets.Dataset.from_dict({self.identifier: data}, features=self.features)


class ThresholdLogitsProcessor(TopKLogitsProcessor):

    def __init__(self, head:PredictionHead, t:float) -> None:
        super(ThresholdLogitsProcessor, self).__init__(head, k=None)
        self.t = t

    @property
    def identifier(self) -> str:
        return "t=%.02f" % self.t

    def preprocess(self, logits:torch.Tensor, labels:torch.Tensor) -> Any:
        return torch.sigmoid(logits) >= self.t
