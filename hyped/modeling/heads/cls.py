from transformers import PreTrainedModel
from transformers.adapters.heads import ClassificationHead
from .base import HypedPredictionHead, HypedPredictionHeadConfig

from datasets.features import ClassLabel
from dataclasses import dataclass
from typing import Literal

@dataclass
class HypedClsHeadConfig(HypedPredictionHeadConfig):
    head_type:Literal['hyped-cls-head'] = 'hyped-cls-head'

    layers:int = 2
    activation_function:str = "tanh"
    use_pooler:bool = False
    bias:bool = True

    def get_label_space(self, feature):
        if not isinstance(feature, ClassLabel):
            raise ValueError("Expected label feature for text classification to be `ClassLabel`, got %s." % str(feature))
        # return label space
        return feature.names


class HypedClsHead(HypedPredictionHead, ClassificationHead):

    def __init__(
        self,
        model:PreTrainedModel,
        head_name:str,
        label_column:str,
        loss_coeff:float = 1.0,
        **kwargs
    ) -> None:
        # initialize base classes in correct order
        ClassificationHead.__init__(
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
    wrapped_forward = ClassificationHead.forward
