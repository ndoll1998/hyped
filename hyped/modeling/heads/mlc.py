import torch
from transformers import PreTrainedModel
from transformers.adapters.heads import MultiLabelClassificationHead
from .base import HypedPredictionHead, HypedPredictionHeadConfig

from datasets.features import Sequence, ClassLabel
from dataclasses import dataclass
from typing import Literal

@dataclass
class HypedMlcHeadConfig(HypedPredictionHeadConfig):
    head_type:Literal['hyped-mlc-head'] = 'hyped-mlc-head'

    layers:int = 2
    activation_function:str = "tanh"
    use_pooler:bool = False
    bias:bool = True

    def get_label_space(self, feature):
        if not isinstance(feature, Sequence) and isinstance(feature.feature, ClassLabel):
            raise ValueError("Expected label feature for multi-label classification to be a `Sequence` of `ClassLabel`, got %s." % str(feature))
        # return label space
        return feature.feature.names

class HypedMlcHead(HypedPredictionHead, MultiLabelClassificationHead):

    def __init__(
        self,
        model:PreTrainedModel,
        head_name:str,
        label_column:str,
        loss_coeff:float = 1.0,
        **kwargs
    ) -> None:
        # initialize base classes in correct order
        MultiLabelClassificationHead.__init__(
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
    wrapped_forward = MultiLabelClassificationHead.forward

    def collate_labels(self, labels:list[list[int]], return_tensors:str ='pt'):
        if return_tensors != 'pt':
            raise NotImplementedError()

        num_labels = self.config['num_labels']
        # binarize labels
        binarized_labels = torch.zeros((len(labels), num_labels), dtype=int)
        for i, ids in enumerate(labels):
            binarized_labels[i, ids] = 1
        # return
        return binarized_labels
