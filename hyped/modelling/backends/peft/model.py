from dataclasses import dataclass
from typing import Literal

from datasets import Features
from peft import get_peft_model

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

    def prepare(self, features: Features) -> None:
        # prepare the base model configuration and
        # the peft configuration
        super(PeftModelConfig, self).prepare(features)
        self.peft_config.prepare(self.head)


class PeftModel(HuggingFaceModel):
    # overwrit config type
    CONFIG_TYPE = PeftModelConfig

    def __init__(self, config: PeftModelConfig) -> None:
        # initialize as a huggingface model and convert to peft model
        super(PeftModel, self).__init__(config)
        self.model = get_peft_model(self.model, self.config.peft_config)
