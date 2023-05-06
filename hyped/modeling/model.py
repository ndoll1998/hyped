from __future__ import annotations

import torch
import torch.nn as nn

import copy
import logging

from transformers import (
    AutoModel,
    AutoConfig,
    PretrainedConfig,
    PreTrainedModel
)

logger = logging.getLogger(__name__)

# TODO: head configs need to be registered at AutoConfig
# TODO: head models need to be registered at AutoModel
# TODO: head config gets encoder config at init to extract hidden dim and whatever

class ArbitraryEncoderPlusHeadsConfig(PretrainedConfig):
    model_type = "encoder-plus-heads"
    is_composition = True

    def __init__(self, encoder:dict, heads:list[dict] =[], **kwargs) -> None:
        super(ArbitraryEncoderPlusHeadsConfig, self).__init__(**kwargs)
        # create encoder config
        encoder_type = encoder.pop('model_type')
        self.encoder = AutoConfig.for_model(encoder_type, **encoder)
        # create all head configurations
        self.heads = []
        for head_c in heads:
            head_t = head_c.pop('model_type')
            head = AutoConfig.for_model(head_t, encoder_config=self.encoder, **head_c)
            self.heads.append(head)

    @classmethod
    def from_configs(
        cls,
        encoder:PretrainedConfig,
        heads:list[PretrainedConfig],
        **kwargs
    ) -> ArbitraryEncoderPlusHeadsConfig:
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

class ArbitraryEncoderPlusHeads(PreTrainedModel):

    config_class = ArbitraryEncoderPlusHeadsConfig
    _keys_to_ignore_on_load_missing = ["heads"]

    def __init__(self, config:ArbitraryEncoderPlusHeadsConfig) -> None:
        # initialize pretrained model
        super(ArbitraryEncoderPlusHeads, self).__init__(config)

        # create encoder and save it using the corresponding base model prefix
        # this is necessary to ensure that `from_pretrained_encoder` loads all
        # pretrained weights correctly
        encoder = AutoModel.from_config(config.encoder)
        self.encoder_attribute_name = encoder.__class__.base_model_prefix
        setattr(self, self.encoder_attribute_name, encoder)
        # create prediction heads
        self.heads = nn.ModuleList([AutoModel.from_config(c) for c in config.heads])

        # check encoder config
        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning("Config of the encoder (%s) is overritten by shared encoder config (%s)" % (self.encoder.__class__, self.config.encoder))
            self.encoder.config = self.config.encoder

        # check head configs
        for i, head_config in enumerate(self.config.heads):
            if self.heads[i].config.to_dict() != head_config.to_dict():
                logger.warning("Config of the %i-th head (%s) is overritten by shared head config (%s)" % (i+1, self.heads[i].__class__, head_config))
                self.heads[i].config = head_config

    @property
    def encoder(self) -> PreTrainedModel:
        return getattr(self, self.encoder_attribute_name)

    @classmethod
    def from_pretrained_encoder(cls, pretrained_model_name_or_path:str, heads=[], **kwargs) -> ArbitraryEncoderPlusHeads:

        # separate kwargs specific for encoder from general model kwargs
        encoder_kwargs = {arg[8:]:kwargs.pop(args) for arg in kwargs if arg.startswith("encoder_")}
        # create config
        config = ArbitraryEncoderPlusHeadsConfig.from_configs(
            encoder=AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                **encoder_kwargs
            ),
            heads=heads,
            **kwargs
        )

        return ArbitraryEncoderPlusHeads.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **kwargs
        )
