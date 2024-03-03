from abc import ABC
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    DefaultDataCollator,
)

from hyped.base.config import BaseConfig
from hyped.modelling.heads import (
    BaseHeadConfig,
    ClassificationHeadConfig,
    TaggingHeadConfig,
)
from hyped.utils.feature_access import FeatureKey, pop_value_at_key


@dataclass
class BaseHuggingFaceDataCollator(BaseConfig, ABC):
    """Abstract base class for HuggingFace Data Collators"""

    t: Literal[
        "hyped.modelling.backends.hf.collator.base"
    ] = "hyped.modelling.backends.hf.collator.base"

    _supported_head_types: ClassVar[tuple[type[BaseHeadConfig]]] = []


@dataclass
class BaseHuggingFaceDataCollatorWithTokenizer(
    BaseHuggingFaceDataCollator, ABC
):
    """Abstract base class for HuggingFace Data Collators that require
    a tokenizer.

    Povides functionality to instantiate the tokenizer given a
    pretrained checkpoint using the `transformers.AutoTokenizer`
    class.
    """

    t: Literal[
        "hyped.modelling.backends.hf.collator.base_with_tokenizer"
    ] = "hyped.modelling.backends.hf.collator.base_with_tokenizer"

    tokenizer: str | PreTrainedTokenizerBase = None

    def __post_init__(self) -> None:
        if isinstance(self.tokenizer, str):
            # load tokenizer from pretrained checkpoint
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer)


@dataclass
class HuggingFaceDefaultDataCollator(
    BaseHuggingFaceDataCollator, DefaultDataCollator
):
    t: Literal[
        "hyped.modelling.backends.hf.collator.default"
    ] = "hyped.modelling.backends.hf.collator.default"

    _supported_head_types = (ClassificationHeadConfig, TaggingHeadConfig)


@dataclass
class HuggingFaceDataCollatorWithPadding(
    BaseHuggingFaceDataCollatorWithTokenizer, DataCollatorWithPadding
):
    t: Literal[
        "hyped.modelling.backends.hf.collator.padding"
    ] = "hyped.modelling.backends.hf.collator.padding"

    _supported_head_types = (ClassificationHeadConfig,)


class HuggingFaceDataCollatorForTagging(
    BaseHuggingFaceDataCollatorWithTokenizer,
    DataCollatorForTokenClassification,
):
    t: Literal[
        "hyped.modelling.backends.hf.collator.tagging"
    ] = "hyped.modelling.backends.hf.collator.tagging"

    _supported_head_types = (TaggingHeadConfig,)


@dataclass
class HuggingFaceMapKeysDataCollatorWrapper(object):
    """Data Collator Wrapper mapping source features to target keys

    Attributes:
        collator (BaseHuggingFaceDataCollator): base data collator to apply
        key_mapping (dict[str, FeatureKey]): feature key mapping
    """

    collator: BaseHuggingFaceDataCollator = field(
        default_factory=HuggingFaceDefaultDataCollator
    )
    key_mapping: dict[str, FeatureKey] = field(default_factory=dict)

    def __call__(self, features, *args, **kwargs) -> dict[str, Any]:
        # map feature keys according to key mapping
        for example in features:
            for tgt, src in self.key_mapping.items():
                example[tgt] = pop_value_at_key(example, src)
        # apply base collator
        return self.collator(features, *args, **kwargs)
