from transformers import PreTrainedModel
from transformers.adapters.heads import CausalLMHead
from .base import HypedPredictionHead, HypedPredictionHeadConfig
from .tagging import HypedTaggingHead

from datasets import Features
from dataclasses import dataclass
from typing import Literal

@dataclass
class HypedCausalLMHeadConfig(HypedPredictionHeadConfig):
    head_type:Literal['hyped-clm-head'] = 'hyped-clm-head'

    layers:int = 1
    activation_function:str = "tanh"
    layer_norm:bool = False
    bias:bool = True
    shift_labels:bool = True

    # default behavior is to reproduce input ids
    # note that shift labels is set to true by default
    label_column:str ="input_ids"

    def __post_init__(self):
        if (not self.shift_labels) and (self.label_column == "input_ids"):
            warnings.warn("Causal LM head got label_column='input_ids' and shift_labels=False. This specifies the trivial task of reproducing the input, NOT next word prediction.", UserWarning)

    def check_and_prepare(self, features:Features):
        # check if labels feature is present
        if self.label_column not in features:
            raise KeyError('Label column `%s` not present in features: %s' % (self.label_column, list(features.keys())))

    def get_label_space(self, feature):
        pass

class HypedCausalLMHead(HypedTaggingHead, CausalLMHead):

    def __init__(
        self,
        model:PreTrainedModel,
        head_name:str,
        label_column:str,
        loss_coeff:float = 1.0,
        **kwargs
    ) -> None:
        # remove label arguments as these are inferred from
        # the model's vocabulary
        kwargs.pop("num_labels", None)
        kwargs.pop("id2label", None)

        # initialize base classes in correct order
        CausalLMHead.__init__(
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
    wrapped_forward = CausalLMHead.forward
