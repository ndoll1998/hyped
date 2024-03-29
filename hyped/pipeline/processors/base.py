from abc import ABC, abstractmethod
from datasets import Features

from inspect import signature
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from typing import Any, Literal

@dataclass
class DataProcessorConfig(object):
    processor_type:Literal['abstract-data-processor'] = 'abstract-data-processor'

class DataProcessor(ABC):
    """Abstract Data Processor"""

    def __init__(self, config:DataProcessorConfig) -> None:
        self._config = config
        self._in_features:Features = None
        self._new_features:Features = None

    @property
    def config(self) -> DataProcessorConfig:
        return self._config

    @property
    def is_prepared(self) -> bool:
        return (self._in_features is not None) and (self._new_features is not None)

    @property
    def in_features(self) -> Features:
        # check if data processor is prepared
        if not self.is_prepared:
            raise RuntimeError("Data processor not prepared. Did you forget to call `prepare` before execution?")
        # return features
        return self._in_features

    @property
    def new_features(self) -> Features:
        # check if data processor is prepared
        if not self.is_prepared:
            raise RuntimeError("Data processor not prepared. Did you forget to call `prepare` before execution?")
        # return features
        return self._new_features

    @property
    def out_features(self) -> Features:
        return Features(self.in_features | self.new_features)

    def prepare(self, features:Features) -> Features:
        # check if data processor is already prepared
        if self.is_prepared:
            raise RuntimeError("Data processor already prepared!")
        # map input features to output features
        # copy as preparation might disturb features inplace
        new_features = self.map_features(features.copy())
        # set features
        self._in_features = features
        self._new_features = new_features
        # return output features
        return self.out_features

    @property
    def is_batched(self) -> bool:
        return 'examples' in signature(self.process).parameters

    @property
    def requires_rank(self) -> bool:
        return 'rank' in signature(self.process).parameters

    @property
    def requires_index(self) -> bool:
        return 'index' in signature(self.process).parameters

    @abstractmethod
    def map_features(self, features:Features) -> Features:
        """ Map input features to *new* features. This specifies the exact output of the `process` function."""
        ...

    @abstractmethod
    def process(self, example:dict[str, Any]) -> dict[str, Any]:
        ...
    @abstractmethod
    def process(self, example:dict[str, Any], rank:int) -> dict[str, Any]:
        ...
    @abstractmethod
    def process(self, example:dict[str, Any], index:int) -> dict[str, Any]:
        ...
    @abstractmethod
    def process(self, example:dict[str, Any], index:int, rank:int) -> dict[str, Any]:
        ...
    @abstractmethod
    def process(self, examples:dict[str, list[Any]]) -> dict[str, list[Any]]:
        ...
    @abstractmethod
    def process(self, examples:dict[str, list[Any]], rank:int) -> dict[str, list[Any]]:
        ...
    @abstractmethod
    def process(self, examples:dict[str, list[Any]], index:int) -> dict[str, list[Any]]:
        ...
    @abstractmethod
    def process(self, examples:dict[str, list[Any]], index:int, rank:int) -> dict[str, list[Any]]:
        ...

    def __call__(
        self, examples:dict[str, list[Any]], index:None|list[int] = None, rank:None|int = None
    ) -> dict[str, list[Any]]:

        if self.is_batched:
            # data processor expects batch of examples
            kwargs = (
                ({'rank': rank} if self.requires_rank else {}) |
                ({'index': index} if self.requires_index else {})
            )
            return examples | self.process(examples, **kwargs)

        processed_examples = defaultdict(list)
        # get the batch size
        n = len(next(iter(examples.values())))
        # apply processor to each item in the batch seperately
        for i in range(n):
            # extract single example from batch
            example = {k: v[i] for k, v in examples.items()}
            # build additional keyword arguments
            kwargs = (
                ({'rank': rank} if self.requires_rank else {}) |
                ({'index': index[i]} if self.requires_index else {})
            )
            # collect all processed examples
            for k, v in self.process(example, **kwargs).items():
                processed_examples[k].append(v)

        # merge and return
        return examples | processed_examples
