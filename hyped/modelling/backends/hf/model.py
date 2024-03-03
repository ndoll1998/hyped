from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from datasets import Features, Value
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from hyped.modelling.backends.base import BaseModel, BaseModelConfig
from hyped.modelling.heads import ClassificationHeadConfig, TaggingHeadConfig
from hyped.utils.feature_access import get_feature_at_key
from hyped.utils.feature_checks import (
    INT_TYPES,
    UINT_TYPES,
    check_feature_exists,
    raise_feature_is_sequence,
)

# map head config type to the corresponding auto model type
HEAD_TO_MODEL_MAPPING = {
    ClassificationHeadConfig: AutoModelForSequenceClassification,
    TaggingHeadConfig: AutoModelForTokenClassification,
}


@dataclass
class HuggingFaceModelConfig(BaseModelConfig):
    """HuggingFace pretrainer Model Configuration

    Attributes:
        head (BaseHeadConfig): the task-specific head to use for the model
        pretrained_ckpt (str): the pretrained checkpoint to load
    """

    t: Literal[
        "hyped.modelling.backends.hf.model"
    ] = "hyped.modelling.backends.hf.model"

    pretrained_ckpt: str = "bert-base-uncased"

    def check_config(self, features: Features) -> None:
        # check required input features
        raise_feature_is_sequence(
            get_feature_at_key(features, "input_ids"), INT_TYPES + UINT_TYPES
        )
        # check optional input features
        if check_feature_exists("attention_mask", features):
            raise_feature_is_sequence(
                get_feature_at_key(features, "attention_mask"),
                [Value("bool")] + INT_TYPES + UINT_TYPES,
            )
        if check_feature_exists("token_type_ids", features):
            raise_feature_is_sequence(
                get_feature_at_key(features, "token_type_ids"),
                INT_TYPES + UINT_TYPES,
            )
        # TODO: check model specific input features like bounding boxes
        # prepare head config
        self.head.prepare(features)


class HuggingFaceModel(BaseModel[HuggingFaceModelConfig]):
    def __init__(self, config: HuggingFaceModelConfig) -> None:
        super(HuggingFaceModel, self).__init__(config)

        # check if the head type is supported
        if type(config.head) not in HEAD_TO_MODEL_MAPPING:
            raise TypeError(
                "Head type `%s` not supported by HuggingFace models"
                % type(config.head).__name__
            )

        # load pretrained model instance
        auto_model_class = HEAD_TO_MODEL_MAPPING[type(config.head)]
        self.model = auto_model_class.from_pretrained(
            config.pretrained_ckpt,
            # specify label space
            num_labels=config.head.num_labels,
            label2id={
                label: i for i, label in enumerate(config.head.label_space)
            },  # noqa: E501
            id2label={
                i: label for i, label in enumerate(config.head.label_space)
            },  # noqa: E501
        )

    @classmethod
    def from_config(cls, config: HuggingFaceModelConfig) -> HuggingFaceModel:
        return cls(config)
