from __future__ import annotations
from abc import ABC, abstractmethod
from datasets import Features
from dataclasses import dataclass
from collections import defaultdict
from hyped.base.config import BaseConfig, BaseConfigurable
from typing import Literal, Any, TypeVar


@dataclass
class BaseDataProcessorConfig(BaseConfig):
    """Base Data Processor Config"""

    t: Literal["hyped.data.processor.base"] = "hyped.data.processor.base"


T = TypeVar("T", bound=BaseDataProcessorConfig)


class BaseDataProcessor(BaseConfigurable[T], ABC):
    """Abstract Base Data Processor

    Provides basic functionality of a data-processor. Sub-types need to
    specify the `process` and `map_features` function.

    Arguments:
        config (BaseDataProcessorConfig): data processor configuration
    """

    @classmethod
    def from_config(cls, config: BaseDataProcessorConfig) -> BaseDataProcessor:
        return cls(config)

    def __init__(self, config: BaseDataProcessorConfig) -> None:
        self._config = config
        self._in_features: Features = None
        self._new_features: Features = None

    @property
    def config(self) -> BaseDataProcessorConfig:
        """Get the processor configuration

        Returns:
            config (BaseDataProcessorConfig): config
        """
        return self._config

    @property
    def is_prepared(self) -> bool:
        """Check if the processor is prepared and ready for execution

        Returns:
            is_prepared (bool): boolean indicating if the processor is prepared
        """
        return (self._in_features is not None) and (
            self._new_features is not None
        )

    def prepare(self, features: Features) -> Features:
        """Prepare the processor for execution

        Arguments:
            features (Features):
                input dataset features available to the processor on execution

        Returns:
            out_features (Features):
                dataset features of the output of the processor
        """
        # map input features to output features
        # copy as preparation might disturb features inplace
        new_features = self.map_features(features.copy())
        # set features
        self._in_features = features
        self._new_features = new_features
        # return output features
        return self.out_features

    @property
    def in_features(self) -> Features:
        """Input dataset features available to processor

        Returns:
            features (Features): input dataset features
        """
        # check if data processor is prepared
        if not self.is_prepared:
            raise RuntimeError(
                "Data processor not prepared. Did you forget to "
                "call `prepare` before execution?"
            )
        # return features
        return self._in_features

    @property
    def new_features(self) -> Features:
        """New dataset features generated by the processor

        Returns:
            features (Features): new dataset features
        """
        # check if data processor is prepared
        if not self.is_prepared:
            raise RuntimeError(
                "Data processor not prepared. Did you forget to "
                "call `prepare` before execution?"
            )
        # return features
        return self._new_features

    @property
    def out_features(self) -> Features:
        """All output features of the processor. Includes both input
        features and new features generated by the processor. On conflicts,
        the new features are prioritized.

        Returns:
            features (Features): complete output dataset features
        """
        return Features(self.in_features | self.new_features)

    def batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> dict[str, list[Any]]:
        """Process a batch of examples

        Arguments:
            examples (dict[str, list[Any]]): batch of examples to process
            index (list[int]): dataset indices of the examples
            rank (int): execution process rank

        Returns:
            out (dict[str, list[Any]]): processed examples
        """
        out = defaultdict(list)
        # process each example one-by-one
        for j, i in enumerate(index):
            example = {k: v[j] for k, v in examples.items()}
            for k, v in self.process(example, index=i, rank=rank).items():
                out[k].append(v)
        # update examples
        return examples | out

    @abstractmethod
    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Abstract process method. Needs to be overwritten in sub-classes.

        Arguments:
            example (dict[str, Any]): example to process
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]): processed example
        """
        ...

    @abstractmethod
    def map_features(self, features: Features) -> Features:
        """Map input features to *new* features. This specifies the exact
        output of the `process` function.

        Arguments:
            in (features): input dataset features

        Returns:
            out (features):
                new dataset features. Note that this is not the output feature
                map but only the features generated by the data processor
        """
        ...
