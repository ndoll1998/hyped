from abc import ABC, abstractmethod

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

    @property
    def requires_rank(self) -> bool:
        return 'rank' in signature(self.process).parameters

    @property
    def requires_index(self) -> bool:
        return 'index' in signature(self.process).parameters

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
