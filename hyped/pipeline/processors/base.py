from abc import ABC, abstractmethod
from datasets import Features

from inspect import signature
from dataclasses import dataclass

import numpy as np
from typing import Any, Literal

@dataclass
class DataProcessorConfig(object):
    processor_type:Literal['abstract-data-processor'] = 'abstract-data-processor'

class DataProcessor(ABC):
    """Abstract Data Processor"""

    def __init__(self, config:DataProcessorConfig) -> None:
        self.config = config
        self._in_features:Features = None
        self._out_features:Features = None

    @property
    def is_prepared(self) -> bool:
        return (self._in_features is not None) and (self._out_features is not None)

    @property
    def features(self) -> Features:
        # check if data processor is prepared
        if not self.is_prepared:
            raise RuntimeError("Data processor not prepared. Did you forget to call `prepare` before execution?")
        # return features
        return self._in_features

    @property
    def out_features(self) -> Features:
        # check if data processor is prepared
        if not self.is_prepared:
            raise RuntimeError("Data processor not prepared. Did you forget to call `prepare` before execution?")
        # return features
        return self._out_features

    def prepare(self, features:Features) -> Features:
        # check if data processor is already prepared
        if self.is_prepared:
            raise RuntimeError("Data processor already prepared!")
        # map input features to output features
        # copy as preparation might disturb features inplace
        out_features = self.map_features(features.copy())
        # set features
        self._in_features = features
        self._out_features = out_features
        # return output features
        return self.out_features

    @property
    def requires_rank(self) -> bool:
        return 'rank' in signature(self.process).parameters

    @property
    def requires_index(self) -> bool:
        return 'index' in signature(self.process).parameters

    @abstractmethod
    def map_features(self, features:Features) -> Features:
        ...

    @abstractmethod
    def process(self, example:Any) -> dict[str, np.ndarray]:
        ...
    @abstractmethod
    def process(self, example:Any, rank:int) -> dict[str, np.ndarray]:
        ...
    @abstractmethod
    def process(self, example:Any, index:int) -> dict[str, np.ndarray]:
        ...
    @abstractmethod
    def process(self, example:Any, index:int, rank:int) -> dict[str, np.ndarray]:
        ...

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)
