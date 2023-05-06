import torch
import torch.nn as nn

from transformers import (
    PretrainedConfig,
    PreTrainedModel
)

class TextClassificationHeadConfig(PretrainedConfig):
    model_type = "seq-cls-head"

    def __init__(
        self,
        dropout:float =0.2,
        **kwargs
    ) -> None:
        super(TextClassificationHeadConfig, self).__init__(**kwargs)
        # save dropout probability
        self.dropout = dropout
        # check if number of labels is set
        if self.num_labels is None:
            raise ValueError("`num_labels` not set")

class TextClassificationHead(PreTrainedModel):

    config_class = TextClassificationHeadConfig
    _keys_to_ignore_on_load_missing = ["classifier"]

    def __init__(
        self,
        config:TextClassificationHeadConfig,
        encoder_config:PretrainedConfig
    ) -> None:
        super(TextClassificationHead, self).__init__(config)
        # create dropout classification head
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(
            encoder_config.hidden_size,
            config.num_labels
        )

