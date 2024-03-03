from dataclasses import dataclass, field
from typing import Literal

import peft

from hyped.base.config import BaseConfig
from hyped.modelling.heads import (
    BaseHeadConfig,
    CausalLanguageModellingHeadConfig,
    ClassificationHeadConfig,
    TaggingHeadConfig,
)

# map head type to the corresponding peft task type
HEAD_TO_TASK_MAPPING = {
    CausalLanguageModellingHeadConfig: peft.TaskType.CAUSAL_LM,
    ClassificationHeadConfig: peft.TaskType.SEQ_CLS,
    TaggingHeadConfig: peft.TaskType.TOKEN_CLS,
}


@dataclass
class BasePeftConfig(BaseConfig):
    t: Literal[
        "hyped.modelling.backends.peft.config.base"
    ] = "hyped.modelling.backends.peft.config.base"

    # the task type is inferred from the head
    task_type: peft.TaskType = field(init=False)

    def prepare(self, head: BaseHeadConfig) -> None:
        """Infer the task type from the given head"""

        if type(head) not in HEAD_TO_TASK_MAPPING:
            raise ValueError(
                "Head type `%s` not supported by peft models"
                % type(head).__name__
            )
        # set the task type according to the head type
        self.task_type = HEAD_TO_TASK_MAPPING[type(head)]


@dataclass
class AdaptionPromptConfig(BasePeftConfig, peft.AdaptionPromptConfig):
    t: Literal[
        "hyped.modelling.backends.peft.config.adaption_prompt"
    ] = "hyped.modelling.backends.peft.config.adaption_prompt"


@dataclass
class LoraConfig(BasePeftConfig, peft.LoraConfig):
    t: Literal[
        "hyped.modelling.backends.peft.config.lora"
    ] = "hyped.modelling.backends.peft.config.lora"


@dataclass
class LoftQConfig(BasePeftConfig, peft.LoftQConfig):
    t: Literal[
        "hyped.modelling.backends.peft.config.loft_q"
    ] = "hyped.modelling.backends.peft.config.loft_q"


@dataclass
class LoHaConfig(BasePeftConfig, peft.LoHaConfig):
    t: Literal[
        "hyped.modelling.backends.peft.config.loha"
    ] = "hyped.modelling.backends.peft.config.loha"


@dataclass
class LoKrConfig(BasePeftConfig, peft.LoKrConfig):
    t: Literal[
        "hyped.modelling.backends.peft.config.lokr"
    ] = "hyped.modelling.backends.peft.config.lokr"


@dataclass
class IA3Config(BasePeftConfig, peft.IA3Config):
    t: Literal[
        "hyped.modelling.backends.peft.config.ia3"
    ] = "hyped.modelling.backends.peft.config.ia3"


@dataclass
class AdaLoraConfig(BasePeftConfig, peft.AdaLoraConfig):
    t: Literal[
        "hyped.modelling.backends.peft.config.ada_lora"
    ] = "hyped.modelling.backends.peft.config.ada_lora"
