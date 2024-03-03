from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

from datasets import Features, Value
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from hyped.modelling.backends.base import BaseModel, BaseModelConfig
from hyped.modelling.heads import (
    CausalLanguageModellingHeadConfig,
    ClassificationHeadConfig,
    TaggingHeadConfig,
)
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
    CausalLanguageModellingHeadConfig: AutoModelForCausalLM,
}


@dataclass
class HuggingFaceModelConfig(BaseModelConfig):
    """HuggingFace pretrainer Model Configuration

    Attributes:
        head (BaseHeadConfig): the task-specific head to use for the model
        pretrained_ckpt (str): the pretrained checkpoint to load
        kwargs (dict[str, Any]):
            keyword arguments forwarded to the `AutoModel.from_pretrained`
            function. Please refer to the transformers documentation for
            more information.
    """

    t: Literal[
        "hyped.modelling.backends.hf.model"
    ] = "hyped.modelling.backends.hf.model"

    pretrained_ckpt: str = "bert-base-uncased"
    kwargs: dict[str, Any] = field(default_factory=dict)

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
    def _build_kwargs(self) -> dict[str, Any]:
        # get the keyword arguments from the configuration
        kwargs = self.config.kwargs.copy()

        # check if user provided information on label space
        if (kwargs.get("num_labels", None) is not None) and (
            kwargs["num_labels"] != self.config.head.num_labels
        ):
            warnings.warn(
                "Model argument `num_labels` (%i) will be overwritten with "
                "inferred value (%i)"
                % (kwargs["num_labels"], self.config.head.num_labels),
                UserWarning,
            )
        # set number of labels in model configuration
        kwargs["num_labels"] = self.config.head.num_labels

        if self.config.head.label_space is not None:
            if kwargs.get("label2id", None) is not None:
                warnings.warn(
                    "Model argument `label2id` will be overwritten with "
                    "inferred value",
                    UserWarning,
                )

            if kwargs.get("id2label", None) is not None:
                warnings.warn(
                    "Model argument `id2label` will be overwritten with "
                    "inferred value",
                    UserWarning,
                )
            # set label mappings
            kwargs["label2id"] = {
                label: i
                for i, label in enumerate(self.config.head.label_space)
            }
            kwargs["id2label"] = {
                i: label
                for i, label in enumerate(self.config.head.label_space)
            }

        return kwargs

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
            config.pretrained_ckpt, **self._build_kwargs()
        )

    @classmethod
    def from_config(cls, config: HuggingFaceModelConfig) -> HuggingFaceModel:
        return cls(config)
