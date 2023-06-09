import hyped
import pydantic
import dataclasses
import datasets
import transformers
from typing import Literal
from typing_extensions import Annotated
from .base import ModelConfig

@dataclasses.dataclass
class ClsHeadConfig(hyped.modeling.adapters.heads.HypedAdapterClsHeadConfig):
    head_type:Literal["classification"] = "classification"

@dataclasses.dataclass
class MlcHeadConfig(hyped.modeling.adapters.heads.HypedAdapterMlcHeadConfig):
    head_type:Literal["multi-label-classification"] = "multi-label-classification"

@dataclasses.dataclass
class TaggingHeadConfig(hyped.modeling.adapters.heads.HypedAdapterTaggingHeadConfig):
    head_type:Literal["tagging"] = "tagging"

@dataclasses.dataclass
class CausalLMHeadConfig(hyped.modeling.adapters.heads.HypedAdapterCausalLMHeadConfig):
    head_type:Literal["causal-lm"] = "causal-lm"

class AdapterTransformerModelConfig(ModelConfig):
    """Adapter Transformer Model Configuration Model"""
    library:Literal['adapter-transformers'] = 'adapter-transformers'
    # adapter setup
    adapter_name:None|str = None # defaults to dataset name
    adapter:None|transformers.adapters.AdapterArguments = None
    # prediction heads
    heads:dict[
        str,
        Annotated[
            (
                ClsHeadConfig |
                MlcHeadConfig |
                TaggingHeadConfig |
                CausalLMHeadConfig
            ),
            pydantic.Field(..., discriminator='head_type')
        ]
    ]

    def check_and_prepare(self, features:datasets.Features) -> None:
        [hconfig.check_and_prepare(features) for hconfig in self.heads.values()]

    @pydantic.validator('heads', pre=True)
    def _pass_head_name_to_config(cls, v):
        assert isinstance(v, dict)
        # add head name entry to corresponding head configuration
        for name, config in v.items():
            # check if it matches the head name if already set
            if config.get('head_name', name) != name:
                raise ValueError("Head name mismatch %s!=%s" % (config['head_name'], name))
            # write head name
            config.update({'head_name': name})

        return v

    def build(self, info:datasets.DatasetInfo) -> transformers.PreTrainedModel:
        # set default adapter name and prepare model for dataset
        self.adapter_name = self.adapter_name or info.builder_name
        self.check_and_prepare(info.features)

        # load pretrained model and wrap it
        model = transformers.AutoAdapterModel.from_pretrained(self.pretrained_ckpt, **self.kwargs)
        model = hyped.modeling.adapters.wrapper.HypedAdapterModelWrapper(model)
        # add all heads to model
        for head_config in self.heads.values():
            head = hyped.modeling.adapters.auto.AutoHypedAdapterHead.from_config(model, head_config)
            model.add_prediction_head(head)
        # activate all heads
        model.active_head = list(model.heads.keys())

        # set up adapter
        if self.adapter is not None:
            # check if name is set
            if self.adapter_name is None:
                raise ValueError("`adapter_name` in model configuration not set!")
            # set up adapter
            transformers.adapters.training.setup_adapter_training(
                model=model,
                adapter_args=self.adapter,
                adapter_name=self.adapter_name
            )

        # freeze/unfreeze model parameters
        model.freeze_model(self.freeze)

        # return model instance
        return model

