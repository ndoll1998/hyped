from __future__ import annotations

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel
)

import copy
import logging
from inspect import signature
from functools import cached_property

from transformers.modeling_outputs import ModelOutput
from .heads.head import PredictionHeadOutput, PredictionHeadConfig, PredictionHead

logger = logging.getLogger(__name__)

class MultiHeadModelOutput(ModelOutput):
    # accumulated loss over all prediction heads
    loss:None|torch.FloatTensor = None
    # encoder states
    last_hidden_state:None|torch.FloatTensor =None
    hidden_states:None|tuple[torch.FloatTensor] =None
    attentions:None|torch.FloatTensor =None
    # prediciton head outputs
    head_outputs:None|tuple[PredictionHeadOutput] =None

class ArbitraryEncoderWithHeadsConfig(PretrainedConfig):
    model_type = "encoder-with-heads"
    is_composition = True

    def __init__(self, encoder:dict, heads:list[dict] =[], **kwargs) -> None:
        super(ArbitraryEncoderWithHeadsConfig, self).__init__(**kwargs)
        # create encoder config
        encoder_type = encoder.pop('model_type')
        self.encoder = AutoConfig.for_model(encoder_type, **encoder)
        # create all head configurations
        self.heads = []
        for head_c in heads:
            head_t = head_c.pop('model_type')
            head = AutoConfig.for_model(head_t, **head_c)
            # check type, thus shouldn't be possible given the way
            # the heads are registered to the auto classes
            assert isinstance(head, PredictionHeadConfig), "Prediction head configurations (%s) must inherit from `PredictionHeadConfig` class." % head.__class__.__name__
            # add to list
            self.heads.append(head)

    @classmethod
    def from_configs(
        cls,
        encoder:PretrainedConfig,
        heads:list[PretrainedConfig],
        **kwargs
    ) -> ArbitraryEncoderWithHeadsConfig:
        # convert configs to dicts
        return cls(
            encoder=encoder.to_dict(),
            heads=[c.to_dict() for c in heads],
            **kwargs
        )

    def to_dict(self) -> dict:
        output = copy.deepcopy(self.__dict__)
        output["encoder"] = self.encoder.to_dict()
        output["heads"] = [c.to_dict() for c in self.heads]
        output["model_type"] = self.__class__.model_type
        return output

class ArbitraryEncoderWithHeads(PreTrainedModel):

    config_class = ArbitraryEncoderWithHeadsConfig
    _keys_to_ignore_on_load_missing = ["heads"]

    def __init__(self,
        config:ArbitraryEncoderWithHeadsConfig,
        encoder:None|PreTrainedModel =None,
        **kwargs
    ) -> None:
        # initialize pretrained model
        super(ArbitraryEncoderWithHeads, self).__init__(config)

        # load encoder if not given
        self.encoder = encoder or AutoModel.from_config(config.encoder, **kwargs)
        # check encoder configuration
        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning("Config of the encoder (%s) is overritten by shared encoder config (%s)" % (self.encoder.__class__, self.config.encoder))
            self.encoder.config = self.config.encoder

        # create prediction heads
        self.heads = nn.ModuleList([
            AutoModel.from_config(c, encoder_config=config.encoder)
            for c in config.heads
        ])
        # check head configs
        for i, head_config in enumerate(self.config.heads):
            if not isinstance(self.heads[i], PredictionHead):
                raise ValueError("Prediction head (%s) must inherit from `PredictionHead` class." % self.heads[i].__class__.__name__)
            if self.heads[i].config.to_dict() != head_config.to_dict():
                logger.warning("Config of the %i-th head (%s) is overritten by shared head config (%s)" % (i+1, self.heads[i].__class__, head_config))
                self.heads[i].config = head_config

    @classmethod
    def from_pretrained_encoder(
        cls,
        pretrained_encoder_name_or_path:str,
        heads:list[PretrainedConfig]=[],
        **kwargs
    ) -> ArbitraryEncoderWithHeads:

        # load pretrained encoder
        encoder = AutoModel.from_pretrained(
            pretrained_encoder_name_or_path,
            **kwargs
        )
        # create model config
        config = ArbitraryEncoderWithHeadsConfig.from_configs(
            encoder=encoder.config,
            heads=heads
        )

        # create model
        return ArbitraryEncoderWithHeads(
            config=config,
            encoder=encoder
        )

    @cached_property
    def encoder_forward_arg_names(self) -> tuple[str]:
        return tuple(signature(self.encoder.forward).parameters.keys())

    def forward(self, **kwargs):
        # output specifications
        output_attentions = kwargs.get('output_attentions', None)
        output_hidden_states = kwargs.get('output_hidden_states', None)
        # build encoder keyword arguments
        enc_kwargs = kwargs.copy()
        # encoder should always return everything
        # just in case a head depends on it
        enc_kwargs['return_dict'] = True
        enc_kwargs['output_attentions'] = True
        enc_kwargs['output_hidden_states'] = True
        # remove all keyword arguments that aren't
        # arguments to the forward call of the encoder
        enc_kwargs = {n:enc_kwargs[n] for n in self.encoder_forward_arg_names if n in enc_kwargs}
        # pass through encoder
        h = self.encoder(**enc_kwargs)

        # apply each head and accumulate loss of all heads
        outs = [head(h, labels=kwargs.get(head.config.label_column, None)) for head in self.heads]
        loss = sum(o.loss * head.config.loss_coeff for o, head in zip(outs, self.heads) if o.loss is not None)

        return MultiHeadModelOutput(
            loss=loss,
            # encoder states
            last_hidden_state=h.last_hidden_state,
            hidden_states=h.hidden_states if output_hidden_states else None,
            attentions=h.attentions if output_attentions else None,
            # head outputs
            head_outputs=outs
        )
