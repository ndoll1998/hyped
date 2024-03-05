from dataclasses import dataclass, field
from typing import Any, Literal

from datasets import Features
from peft import get_peft_model
from peft.utils.other import prepare_model_for_kbit_training

from hyped.modelling.backends.hf.model import (
    HuggingFaceModel,
    HuggingFaceModelConfig,
)

from .configs import BasePeftConfig


@dataclass
class PeftModelConfig(HuggingFaceModelConfig):
    t: Literal[
        "hyped.modelling.backends.peft.model"
    ] = "hyped.modelling.backends.peft.model"

    peft_config: BasePeftConfig = None
    # gradient checkpointing
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: dict[str, Any] = field(default_factory=dict)

    def prepare(self, features: Features) -> None:
        # prepare the base model configuration and
        # the peft configuration
        super(PeftModelConfig, self).prepare(features)
        self.peft_config.prepare(self.head)


class PeftModel(HuggingFaceModel):
    # overwrite config type
    CONFIG_TYPE = PeftModelConfig

    def __init__(self, config: PeftModelConfig) -> None:
        # initialize as a huggingface model
        super(PeftModel, self).__init__(config)
        # prepare model for training, this sets up gradient
        # checkpointing and quantization for the model
        # TODO: quantization + gradient checkpointing doesn't seem to work
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs=self.config.gradient_checkpointing_kwargs,
        )
        # convert to peft model
        self.model = get_peft_model(self.model, self.config.peft_config)
