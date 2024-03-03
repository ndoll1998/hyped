from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from datasets import ClassLabel, Features

from hyped.base.config import BaseConfig
from hyped.utils.feature_access import FeatureKey, get_feature_at_key
from hyped.utils.feature_checks import (
    raise_feature_equals,
    raise_feature_is_sequence,
)


@dataclass
class BaseHeadConfig(BaseConfig, ABC):
    """Base Head Config Class"""

    t: Literal["hyped.modelling.heads.head"] = "hyped.modelling.heads.head"

    @property
    @abstractmethod
    def target_keys(self) -> list[FeatureKey]:
        """Abstract property to get the list of target feature keys"""
        ...

    @abstractmethod
    def prepare(self, features: Features) -> None:
        """Abstract method to prepare the head for the dataset features

        Arguments:
            features (Features): dataset features
        """
        ...

    @property
    @abstractmethod
    def label_space(self) -> list[str]:
        """Abstract property for the label space"""
        ...

    @property
    def num_labels(self) -> int:
        """Total number of labels in the label space"""
        return len(self.label_space)


@dataclass
class ClassificationHeadConfig(BaseHeadConfig):
    """Classification Head Config

    Attributes:
        target (FeatureKey):
            dataset feature to use as the target for the classification head
    """

    t: Literal["hyped.modelling.heads.cls"] = "hyped.modelling.heads.cls"

    target: FeatureKey = "label"

    # values extracted in prepare function
    _target_feature: ClassLabel = field(init=False)

    @property
    def target_keys(self) -> list[FeatureKey]:
        return [self.target]

    def prepare(self, features: Features) -> None:
        # get the label feature from the features and
        # make sure its a class label
        self._target_feature = get_feature_at_key(features, self.target)
        raise_feature_equals(self.target, self._target_feature, ClassLabel)

    @property
    def label_space(self) -> list[str]:
        return self._target_feature.names


@dataclass
class TaggingHeadConfig(ClassificationHeadConfig):
    """Tagging Head Config

    Attributes:
        target (FeatureKey):
            dataset feature to use as the target for the tagging head
    """

    t: Literal[
        "hyped.modelling.heads.tagging"
    ] = "hyped.modelling.heads.tagging"

    targets: FeatureKey = "labels"

    @property
    def target_keys(self) -> list[FeatureKey]:
        return [self.targets]

    def prepare(self, features: Features) -> None:
        f = get_feature_at_key(features, self.targets)
        raise_feature_is_sequence(self.targets, f, ClassLabel)
        # get the class label feature
        self._target_feature = f.feature
