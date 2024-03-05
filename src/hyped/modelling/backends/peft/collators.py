from abc import ABC
from dataclasses import dataclass
from typing import Literal

from hyped.modelling.backends.hf.collators import (
    BaseHuggingFaceDataCollator,
    HuggingFaceDataCollatorForLanguageModelling,
    HuggingFaceDataCollatorForTagging,
    HuggingFaceDataCollatorWithPadding,
    HuggingFaceDefaultDataCollator,
)


@dataclass
class BasePeftDataCollator(BaseHuggingFaceDataCollator, ABC):
    """Abstract Base Class for PEFT Data Collators"""


@dataclass
class PeftDefaultDataCollator(
    BasePeftDataCollator, HuggingFaceDefaultDataCollator
):
    t: Literal[
        "hyped.modelling.backends.peft.collator.default"
    ] = "hyped.modelling.backends.peft.collator.default"


@dataclass
class PeftDataCollatorWithPadding(
    BasePeftDataCollator, HuggingFaceDataCollatorWithPadding
):
    t: Literal[
        "hyped.modelling.backends.peft.collator.padding"
    ] = "hyped.modelling.backends.peft.collator.padding"


@dataclass
class PeftDataCollatorForTagging(
    BasePeftDataCollator, HuggingFaceDataCollatorForTagging
):
    t: Literal[
        "hyped.modelling.backends.peft.collator.tagging"
    ] = "hyped.modelling.backends.peft.collator.tagging"


@dataclass
class PeftDataCollatorForLanguageModelling(
    BasePeftDataCollator, HuggingFaceDataCollatorForLanguageModelling
):
    t: Literal[
        "hyped.modelling.backends.peft.collator.language_modelling"
    ] = "hyped.modelling.backends.peft.collator.language_modelling"
