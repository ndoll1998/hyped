from . import heads
from transformers import PreTrainedModel
from transformers.adapters import PredictionHead
from hyped.utils.typedmapping import typedmapping
from functools import cmp_to_key

class AutoHypedAdapterHead(object):

    HEAD_MAPPING = typedmapping[
        type[heads.HypedHeadConfig],
        type[heads.HypedAdapterHead]
    ]()

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated using the `AutoConfig.from_config` method."
        )

    @classmethod
    def from_config(
        cls,
        model:PreTrainedModel,
        config:heads.HypedHeadConfig
    ) -> heads.HypedAdapterHead:
        # find head corresponding to config type
        key = cmp_to_key(lambda t, v: 2 * issubclass(v, t) - 1)
        for config_t in sorted(cls.HEAD_MAPPING, key=key):
            if isinstance(config, config_t):
                return cls.HEAD_MAPPING[config_t](model, config)
        # no head type found for config
        raise ValueError("No head type registered for config of type `%s`." % type(config))

    @classmethod
    def from_head(
        cls,
        model:PreTrainedModel,
        head:PredictionHead,
        **kwargs
    ) -> heads.HypedAdapterHead:
        # find head type corrensponding to head
        key = cmp_to_key(lambda t, v: 2 * issubclass(v[1], t[1]) - 1)
        for config_t, head_t in sorted(cls.HEAD_MAPPING.items(), key=key):
            if issubclass(head_t, type(head)):
                # create head instance and load in parameters from given head
                config = config_t.from_head(head, **kwargs)
                new_head = head_t(model, config)
                new_head.load_state_dict(head.state_dict())
                return new_head
        # no head type found
        raise ValueError("No head type registered for original head of type `%s`." % type(head))

    @classmethod
    def register(cls, config_t:heads.HypedHeadConfig, head_t:heads.HypedAdapterHead) -> None:
        cls.HEAD_MAPPING[config_t] = head_t

AutoHypedAdapterHead.register(heads.HypedAdapterClsHeadConfig, heads.HypedAdapterClsHead)
AutoHypedAdapterHead.register(heads.HypedAdapterMlcHeadConfig, heads.HypedAdapterMlcHead)
AutoHypedAdapterHead.register(heads.HypedAdapterTaggingHeadConfig, heads.HypedAdapterTaggingHead)
AutoHypedAdapterHead.register(heads.HypedAdapterCausalLMHeadConfig, heads.HypedAdapterCausalLMHead)
