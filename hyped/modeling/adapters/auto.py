from . import heads
from transformers import PreTrainedModel
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
    def register(cls, config_t:heads.HypedHeadConfig, head_t:heads.HypedAdapterHead) -> None:
        cls.HEAD_MAPPING[config_t] = head_t

AutoHypedAdapterHead.register(heads.HypedAdapterClsHeadConfig, heads.HypedAdapterClsHead)
AutoHypedAdapterHead.register(heads.HypedAdapterTaggingHeadConfig, heads.HypedAdapterTaggingHead)
