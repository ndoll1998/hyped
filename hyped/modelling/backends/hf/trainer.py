from dataclasses import dataclass, field
from typing import Literal

import transformers
from torch.utils.data import Dataset

from hyped.modelling.backends.base import BaseTrainer, BaseTrainerConfig
from hyped.modelling.backends.hf.collators import (
    BaseHuggingFaceDataCollator,
    HuggingFaceDefaultDataCollator,
    HuggingFaceMapKeysDataCollatorWrapper,
)
from hyped.modelling.backends.hf.model import HuggingFaceModel
from hyped.utils.feature_access import unpack_feature_key


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

    args: transformers.TrainingArguments = None
    collator: BaseHuggingFaceDataCollator = field(
        default_factory=HuggingFaceDefaultDataCollator
    )


class HuggingFaceTrainer(BaseTrainer[BaseTrainerConfig], transformers.Trainer):
    """HuggingFace Trainer"""

    def _build_data_collator(
        self, model: HuggingFaceModel
    ) -> BaseHuggingFaceDataCollator | HuggingFaceMapKeysDataCollatorWrapper:
        """Wrap the data collator to map target keys specified by the head
        to the label keys expected by the model.

        Arguments:
            model (HuggingFaceModel): model to taylor the collator to

        Returns:
            collator (
                BaseHuggingFaceDataCollator
                | HuggingFaceMapKeysDataCollatorWrapper
            ):
                data collator taylored to the given model
        """

        # when the head doesn't specify any labels do nothing
        if len(model.config.head.target_keys) == 0:
            return self.config.collator

        # get the label column of the model
        tgt_keys = transformers.utils.generic.find_labels(
            type(model.model)
        ) or ["labels"]
        src_keys = list(map(unpack_feature_key, model.config.head.target_keys))
        # check that number of keys match up
        if len(tgt_keys) != len(src_keys):
            raise ValueError(
                "Unexpected number of target features, expected %s, got %s"
                % (str(tgt_keys), str(src_keys))
            )

        # find keys that need to be mapped
        remaining_tgt_keys = [k for k in tgt_keys if k not in src_keys]
        remaining_src_keys = [k for k in src_keys if k not in tgt_keys]
        assert len(remaining_tgt_keys) == len(remaining_src_keys)

        if len(remaining_tgt_keys) > 1:
            raise ValueError("Cannot disambiguate target features")

        elif len(remaining_tgt_keys) == 1:
            # map features accordingly
            return HuggingFaceMapKeysDataCollatorWrapper(
                collator=self.config.collator,
                key_mapping={remaining_tgt_keys[0]: remaining_src_keys[0]},
            )

        # use the specified data collator as is
        return self.config.collator

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

        label_names = [
            k[0] if isinstance(k, tuple) else k
            for k in model.config.head.target_keys
        ]
        # set the label names in the trainer configuration
        self.config.args.label_names = self.config.args.label_names or []
        self.config.args.label_names += [
            k for k in label_names if k not in self.config.args.label_names
        ]

        # enable input gradients to allow gradient computations
        # with gradient checkpointing enabled
        if self.config.args.gradient_checkpointing:
            model.model.enable_input_require_grads()

        # initialize trainer
        transformers.Trainer.__init__(
            self,
            model=model.model,
            args=self.config.args,
            data_collator=self._build_data_collator(model),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        # start training
        transformers.Trainer.train(self)
