from transformers import PreTrainedModel
from transformers.adapters.heads import TaggingHead
from .base import HypedPredictionHead, HypedPredictionHeadConfig

from datasets.features import Sequence, ClassLabel
from dataclasses import dataclass
from typing import Literal

@dataclass
class HypedTaggingHeadConfig(HypedPredictionHeadConfig):
    head_type:Literal['hyped-tagging-head'] = 'hyped-tagging-head'

    layers:int = 1
    activation_function:str = "tanh"

    def get_label_space(self, feature):
        if not isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel):
            raise ValueError("Expected label feature for tagging to be a `Sequence` of `ClassLabel`, got %s." % str(feature))
        # return label space
        return feature.feature.names

class HypedTaggingHead(HypedPredictionHead, TaggingHead):

    def __init__(
        self,
        model:PreTrainedModel,
        head_name:str,
        label_column:str,
        loss_coeff:float = 1.0,
        **kwargs
    ) -> None:
        # initialize base classes in correct order
        TaggingHead.__init__(
            self,
            model=model,
            head_name=head_name,
            **kwargs
        )
        HypedPredictionHead.__init__(
            self,
            label_column=label_column,
            loss_coeff=loss_coeff
        )

    # set forward function
    wrapped_forward = TaggingHead.forward
