from abc import ABC, abstractmethod

from inspect import signature
from dataclasses import dataclass

import numpy as np
from typing import Any, Literal


@dataclass
class DataFilterConfig(object):
    filter_type:Literal['abstract-data-filter'] = 'abstract-data-filter'

class DataFilter(ABC):
    """Abstract Data Filter"""

    def __init__(self, config:DataFilterConfig) -> None:
        self.config = config

    @property
    def requires_index(self) -> bool:
        return 'index' in signature(self.filter).parameters

    @abstractmethod
    def filter(self, example:Any) -> bool:
        ...
    @abstractmethod
    def filter(self, example:Any, rank:int) -> bool:
        ...
    @abstractmethod
    def filter(self, example:Any, index:int) -> bool:
        ...
    @abstractmethod
    def filter(self, example:Any, index:int, rank:int) -> bool:
        ...

    def __call__(self, *args, **kwargs):
        return self.filter(*args, **kwargs)
