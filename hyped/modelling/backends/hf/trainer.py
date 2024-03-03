from dataclasses import dataclass
from typing import Literal

from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from hyped.modelling.backends.base import BaseTrainer, BaseTrainerConfig
from hyped.modelling.backends.hf.collators import (
    BaseHuggingFaceDataCollator,
    HuggingFaceDefaultDataCollator,
)
from hyped.modelling.backends.hf.model import HuggingFaceModel


@dataclass
class HuggingFaceTrainerConfig(BaseTrainerConfig):
    """HuggingFace Trainer Configuration

    Attributes:
        args (transformers.TrainingArguments):
            training arguments, see transformers documentation for more
            information
        collator (BaseHuggingFaceDataCollator):
            data collator to use, defaults to the
            `HuggingFaceDefaultDataCollator`
    """

    t: Literal[
        "hyped.modelling.backends.hf.trainer"
    ] = "hyped.modelling.backends.hf.trainer"

    args: TrainingArguments = None
    collator: BaseHuggingFaceDataCollator = HuggingFaceDefaultDataCollator()


class HuggingFaceTrainer(BaseTrainer[BaseTrainerConfig], Trainer):
    """HuggingFace Trainer"""

    def train(
        self,
        model: HuggingFaceModel,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ) -> None:
        # make sure the specified collator supports the head type of the model
        if not isinstance(
            model.config.head, self.config.collator._supported_head_types
        ):
            raise RuntimeError(
                "The specified model head type `%s` is not supported "
                "by the collator `%s`"
                % (
                    type(model.config.head).__name__,
                    type(self.config.collator).__name__,
                )
            )

        # initialize trainer
        Trainer.__init__(
            self,
            model=model.model,
            args=self.config.args,
            data_collator=self.config.collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # start training
        Trainer.train(self)
