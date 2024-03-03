import warnings
from copy import copy
from dataclasses import dataclass
from typing import Literal

from torch.utils.data import Dataset

from hyped.modelling.backends.hf.trainer import (
    HuggingFaceTrainer,
    HuggingFaceTrainerConfig,
)
from hyped.modelling.backends.peft.model import PeftModel


@dataclass
class PeftTrainerConfig(HuggingFaceTrainerConfig):
    t: Literal[
        "hyped.modelling.backends.peft.trainer"
    ] = "hyped.modelling.backends.peft.trainer"

    def __post_init__(self) -> None:
        # check gradient checkpointing value
        if self.args.gradient_checkpointing:
            warnings.warn(
                "Gradient Checkpointing will be overwritten in "
                "TrainingArguments. Please specify the Gradient "
                "Checkpointing parameters for peft models in the "
                "model configuration instead.",
                UserWarning,
            )


class PeftTrainer(HuggingFaceTrainer):
    # overwrite config type
    CONFIG_TYPE = PeftTrainerConfig

    def train(
        self,
        model: PeftModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ) -> None:
        # overwrite gradient checkpointing configuration
        self.config.args = copy(self.config.args)
        self.config.args.gradient_checkpointing = (
            model.config.gradient_checkpointing
        )
        self.config.args.gradient_checkpointing_kwargs = (
            model.config.gradient_checkpointing_kwargs
        )
        # train model
        super(PeftTrainer, self).train(
            model=model, train_dataset=train_dataset, eval_dataset=eval_dataset
        )
