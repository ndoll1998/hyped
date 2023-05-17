from transformers import AutoConfig, PretrainedConfig
from transformers.adapters import AutoAdapterModel
# custom heads
from . import heads
# typing utils
from transformers.adapters import PredictionHead
from hyped.utils.typedmapping import typedmapping

class HypedAutoAdapterModel(AutoAdapterModel):
    CUSTOM_HEAD_MAPPING = typedmapping[str, type[PredictionHead]]()

    @classmethod
    def _register_custom_heads_to_config(cls, config):
        # check for custom heads attribute
        if not hasattr(config, 'custom_heads'):
            config.custom_heads = {}

        for head_name, head_type in cls.CUSTOM_HEAD_MAPPING.items():
            # add custom head to config lookup table if not already present
            # compare to `register_custom_head` in `ModelWithFlexibleHeadsAdaptersMixin`
            if head_name not in config.custom_heads:
                config.custom_heads[head_name] = head_type

        return config

    @classmethod
    def from_config(cls, config):
        # prepare config and initialize model from it
        config = cls._register_custom_heads_to_config(config)
        return super(HypedAutoAdapterModel, cls).from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):

        # check if config if given
        if 'config' in kwargs:
            # get config and check type
            config = kwargs['config']
            if not isinstance(config, PretrainedConfig):
                raise ValueError("Config must be an instance of `PretrainedConfig`, got %s." % type(config))
            # prepare config and write back to keyword args
            kwargs['config'] = cls._register_custom_heads_to_config(config)

        else:
            # load auto pretrained config
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            # apply all keyword args to config, remaining arguments are passed to model
            for key in kwargs.keys():
                if hasattr(config, key):
                    setattr(config, key, kwargs.pop(key))
            # prepare config and write to kwargs
            kwargs['config'] = cls._register_custom_heads_to_config(config)

        # load pretrained model
        return super(HypedAutoAdapterModel, cls).from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs)

    @classmethod
    def register_custom_head(cls, head_name, head_type):
        cls.CUSTOM_HEAD_MAPPING[head_name] = head_type

# register prediction heads
HypedAutoAdapterModel.register_custom_head(heads.HypedClsHeadConfig.head_type, heads.HypedClsHead)
HypedAutoAdapterModel.register_custom_head(heads.HypedMlcHeadConfig.head_type, heads.HypedMlcHead)
HypedAutoAdapterModel.register_custom_head(heads.HypedTaggingHeadConfig.head_type, heads.HypedTaggingHead)
