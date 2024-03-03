from dataclasses import dataclass
from typing import Literal

from hyped.modelling.backends.hf.trainer import (
    HuggingFaceTrainer,
    HuggingFaceTrainerConfig,
)


@dataclass
class PeftTrainerConfig(HuggingFaceTrainerConfig):
    t: Literal[
        "hyped.modelling.backends.peft.trainer"
    ] = "hyped.modelling.backends.peft.trainer"


class PeftTrainer(HuggingFaceTrainer):
    # overwrite config type
    CONFIG_TYPE = PeftTrainerConfig
