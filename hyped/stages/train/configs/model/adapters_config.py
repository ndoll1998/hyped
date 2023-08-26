import os
import hyped
import pydantic
import dataclasses
import datasets
import transformers
from typing import Literal
from typing_extensions import Annotated
from .base import ModelConfig
# import adapters backend
import hyped.modeling.adapters

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
    backend:Literal['adapter-transformers'] = 'adapter-transformers'
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

    @property
    def trainer_t(self) -> transformers.Trainer:
        # use the adapter trainer if the model is froozen
        return transformers.Trainer if not self.freeze else transformers.adapters.AdapterTrainer

    def check_and_prepare(self, features:datasets.Features) -> None:
        [hconfig.check_and_prepare(features) for hconfig in self.heads.values()]

    @pydantic.field_validator('heads', mode='before')
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

    def load(self, ckpt:str) -> transformers.PreTrainedModel:

        if self.freeze:
            # only the adapter and head are saved to the directory
            # the pretrained model has to be loaded from the pretrained ckpt
            model = transformers.AutoAdapterModel.from_pretrained(self.pretrained_ckpt)
            # load adapter
            if self.adapter is not None:
                assert self.adapter_name is not None
                # TODO: identify adapter name from checkpoint directory structure if not set
                model.load_adapter(os.path.join(ckpt, self.adapter_name))
                model.active_adapters = self.adapter_name

            # load all heads
            for head_name in self.heads.keys():
                model.load_head(os.path.join(ckpt, head_name))

        else:
            # the whole model is saved
            model = transformers.AutoAdapterModel.from_pretrained(ckpt)
            # activate adapter if present
            if len(model.config.adapters.adapters) > 0:
                adapter_name = next(iter(model.config.adapters.adapters.keys()))
                adapter_name = self.adapter_name or adapter_name
                model.active_adapters = adapter_name

        # wrap model
        model = hyped.modeling.adapters.HypedAdapterModelWrapper(model)

        for head_name, head in model.heads.items():
            config = self.heads[head_name]
            # the wrapper converts all heads to hyped heads but sets
            # extra arguments to their default value (i.e. label_column, loss_coeff, etc.)
            head = hyped.modeling.adapters.auto.AutoHypedAdapterHead.from_head(
                model, head, loss_coeff=config.loss_coeff, label_column=config.label_column
            )
            # overwrite head
            model.add_prediction_head(head, overwrite_ok=True)

        # activate all heads
        model.active_head = list(model.heads.keys())

        # freeze/unfreeze model parameters
        model.freeze_model(self.freeze)

        return model
