from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, TypeVar

from datasets import Features
from torch.utils.data import Dataset

from hyped.base.config import BaseConfig, BaseConfigurable
from hyped.modelling.heads import BaseHeadConfig


@dataclass
class BaseModelConfig(BaseConfig):
    """Abstract Base Class for Model Configurations

    Subclasses must specify the `check_config` function.

    Attributes:
        head (BaseHeadConfig): the task-specific head to use for the model
    """

    t: Literal[
        "hyped.modelling.backends.model.base"
    ] = "hyped.modelling.backends.model.base"

    head: BaseHeadConfig = None

    def prepare(self, features: Features) -> None:
        """Prepare the model configuration for the dataset features

        Arguments:
            features (Features): dataset features
        """
        # prepare model head for dataset
        self.head.prepare(features)
        self.check_config(features)

    @abstractmethod
    def check_config(self, features: Features) -> None:
        """Abstract function to check the compatibility of the model
        with the dataset features

        Arguments:
            features (Features): dataset features
        """
        ...


T = TypeVar("T", bound=BaseModelConfig)


class BaseModel(BaseConfigurable[T]):
    """Base Model"""

    def __init__(self, config: T) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: T) -> BaseModel:
        return cls(config)


@dataclass
class BaseTrainerConfig(BaseConfig):
    """Base Trainer Configuration"""

    t: Literal[
        "hyped.modelling.backends.trainer.base"
    ] = "hyped.modelling.backends.trainer.base"


V = TypeVar("V", bound=BaseTrainerConfig)


class BaseTrainer(BaseConfigurable[V], ABC):
    """Base Trainer"""

    def __init__(self, config: V) -> None:
        super(BaseTrainer).__init__()
        self.config = config

    @classmethod
    def from_config(cls, config: V) -> BaseTrainer:
        return cls(config)

    @abstractmethod
    def train(
        self,
        model: BaseModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ) -> None:
        """Train a model"""
        ...
