import torch
import torch.nn as nn
import torch.nn.functional as F
# base class
from .head import (
    PredictionHeadOutput,
    PredictionHeadConfig,
    PredictionHead
)
# typing
from transformers import PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from datasets.features import Features, ClassLabel

class TextClsOutput(PredictionHeadOutput):
    ...

class TextClsHeadConfig(PredictionHeadConfig):
    model_type = "text-cls-head"

    def __init__(
        self,
        dropout:float =0.2,
        **kwargs
    ) -> None:
        super(TextClsHeadConfig, self).__init__(**kwargs)
        # save dropout probability
        self.dropout = dropout

    def get_label_space(self, features:Features) -> list[str]:
        # check if label column is present in dataset features
        if self.label_column not in features:
            raise ValueError("Label column `%s` not present in dataset features (%s)." % (
                self.label_column, str(list(features.keys()))))
        if not isinstance(features[self.label_column], ClassLabel):
            raise ValueError("Expected label feature for text classification to be `ClassLabel`, got %s." % str(feature))
        # return label space
        return features[self.label_column].names

class TextClsHead(PredictionHead):

    config_class = TextClsHeadConfig
    _keys_to_ignore_on_load_missing = ["classifier"]

    def __init__(
        self,
        config:TextClsHeadConfig,
        encoder_config:PretrainedConfig
    ) -> None:
        super(TextClsHead, self).__init__(config, encoder_config)
        # check if number of labels is set
        if self.config.num_labels < 0:
            raise ValueError("`num_labels` not set, did you forget to set the label space before initializing the prediction head?")
        # create 2-layer classification network with
        # dropout and tanh activation in between layers
        self.classifier = nn.Sequential(
            nn.Linear(
                encoder_config.hidden_size,
                encoder_config.hidden_size
            ),
            nn.Tanh(),
            nn.Dropout(config.dropout),
            nn.Linear(
                encoder_config.hidden_size,
                config.num_labels
            ),
        )

    def forward(self, encoder_output:ModelOutput, labels:torch.Tensor =None):
        # get last hidden state of first token, should be [CLS]
        # and pass encoding through classifier computing logits
        h = encoder_output.last_hidden_state[:, 0, :]
        logits = self.classifier(h)
        # compute loss from logits it labels are provided
        loss = None if labels is None else \
            F.cross_entropy(
                logits.view(-1, self.config.num_labels),
                labels.view(-1)
            )
        # build output
        return TextClsOutput(
            loss=loss,
            logits=logits
        )
