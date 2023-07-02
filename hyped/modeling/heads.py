from abc import ABC, abstractmethod
from datasets import Features, ClassLabel, Sequence
from dataclasses import dataclass, field
from typing import Any
import warnings

@dataclass
class HypedHeadConfig(ABC):

    head_name:str
    loss_coeff:float = 1.0

    @abstractmethod
    def check_and_prepare(self, features:Features) -> None:
        ...

    @property
    @abstractmethod
    def label_columns(self) -> list[str]:
        ...

@dataclass
class HypedClsHeadConfig(HypedHeadConfig):
    # label column
    label_column:str = 'labels'
    # classification specific configurations
    num_labels:None|int = None
    id2label:None|list[str] = None

    def check_and_prepare(self, features:Features) -> None:

        # check if labels features are present
        if self.label_column not in features:
            raise KeyError("Label column `%s` not present in features %s" % (column, list(features.keys())))

        # get label space from labels feature
        labels_feature = features[self.label_column]
        label_space = self.get_label_space(labels_feature)

        # only set values if extracted label space is valid
        if label_space is not None:
            # warn about overwriting num_labels and id2label
            if (self.num_labels is not None) and (self.num_labels != len(label_space)):
                logger.warn("Overwriting `num_labels` in %s." % type(self))
            if self.id2label is not None:
                logger.warn("Overwriting `id2label` in %s." % type(self))

            # specify label space in config
            self.num_labels = len(label_space)
            self.id2label = dict(enumerate(label_space))

        else:
            # warn
            warnings.warn(
                "Could not extract label space from dataset feature `%s`" % str(labels_feature),
                UserWarning
            )

    def get_label_space(self, feature) -> list[str]:
        if not isinstance(feature, ClassLabel):
            raise ValueError("Expected label feature for text classification to be `ClassLabel`, got %s." % str(feature))
        # return label space
        return feature.names

    @property
    def label_columns(self) -> list[str]:
        return [self.label_column]

@dataclass
class HypedMlcHeadConfig(HypedClsHeadConfig):

    def get_label_space(self, feature) -> list[str]:
        if not (isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel)):
            raise ValueError("Expected label feature for multi-label classification to be a `Sequence` of `ClassLabel`, got %s." % str(feature))
        # return label space
        return feature.feature.names

@dataclass
class HypedTaggingHeadConfig(HypedClsHeadConfig):

    def get_label_space(self, feature) -> list[str]:
        if not (isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel)):
            raise ValueError("Expected label feature for tagging to be a `Sequence` of `ClassLabel`, got %s." % str(feature))
        # return label space
        return feature.feature.names

@dataclass
class HypedCausalLMHeadConfig(HypedClsHeadConfig):
    # default behavior is to reproduce input ids
    # note that shift labels is set to true by default
    label_column:str ="input_ids"

    def get_label_space(self, feature) -> None|list[str]:

        if self.label_column == "input_ids":
            # in this case the column is already encoded and the
            # label space is the vocabulary which we can't access here
            return None

        # otherwise the feature must be a sequence of class labels
        # allowing to extract the label space from it
        if not (isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel)):
            raise ValueError("Expected label feature for causal LM to be a `Sequence` of `ClassLabel`, got %s." % str(feature))
        # return label space
        return feature.feature.names
