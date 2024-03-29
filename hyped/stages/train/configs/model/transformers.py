import hyped
import datasets
import transformers
from enum import Enum
from typing import Literal
from .base import ModelConfig

class Task(Enum):
    CLASSIFICATION              = 'classification'
    MULTI_LABEL_CLASSIFICATION  = 'multi-label-classification'
    TOKEN_CLASSIFICATION        = 'tagging'
    CAUSAL_LANGUAGE_MODELING    = 'causal-language-modeling'

    @property
    def auto_class(self) -> type[transformers.AutoModel]:
        return {
            type(self).CLASSIFICATION:              transformers.AutoModelForSequenceClassification,
            type(self).MULTI_LABEL_CLASSIFICATION:  transformers.AutoModelForSequenceClassification,
            type(self).TOKEN_CLASSIFICATION:        transformers.AutoModelForTokenClassification,
            type(self).CAUSAL_LANGUAGE_MODELING:    transformers.AutoModelForCausalLM
        }[self]

    @property
    def head_config_class(self) -> type[hyped.modeling.heads.HypedHeadConfig]:
        return {
            type(self).CLASSIFICATION:              hyped.modeling.heads.HypedClsHeadConfig,
            type(self).MULTI_LABEL_CLASSIFICATION:  hyped.modeling.heads.HypedMlcHeadConfig,
            type(self).TOKEN_CLASSIFICATION:        hyped.modeling.heads.HypedTaggingHeadConfig,
            type(self).CAUSAL_LANGUAGE_MODELING:    hyped.modeling.heads.HypedCausalLMHeadConfig
        }[self]

    @property
    def problem_type(self) -> str:
        return {
            type(self).CLASSIFICATION:              "single_label_classification",
            type(self).MULTI_LABEL_CLASSIFICATION:  "multi_label_classification",
        }.get(self, None) # return None for other tasks

class TransformerModelConfig(ModelConfig):
    backend:Literal['transformers'] = 'transformers'
    # specify head
    task:Task
    head_name:str
    label_column:None|str = None

    def build(self, info:datasets.DatasetInfo) -> transformers.PreTrainedModel:

        # load pretrained config
        config, kwargs = transformers.AutoConfig.from_pretrained(
            self.pretrained_ckpt,
            **self.kwargs,
            return_unused_kwargs=True
        )
        # set the problem type, important for sequence classification as this argument
        # indicates a single- or multi-label sequence classification task
        # for other tasks this is set to None (see Task.problem_type)
        config.problem_type = self.task.problem_type

        # create the head config
        h_config = self.task.head_config_class(
            head_name=self.head_name,
            label_column=self.label_column
        ) if self.label_column is not None else h.head_config_class(
            head_name=self.head_name
        )

        # prepare head config for dataset
        h_config.check_and_prepare(info.features)

        # update config to match head config
        if h_config.num_labels is not None:
            config.num_labels = h_config.num_labels
        if h_config.id2label is not None:
            config.id2label = h_config.id2label

        # load pretrained model and wrap it
        model = hyped.modeling.transformers.HypedTransformerModelWrapper(
            model=self.task.auto_class.from_pretrained(
                self.pretrained_ckpt,
                config=config,
                **kwargs
            ),
            h_config=h_config
        )

        # freeze/unfreeze pretrained weights
        model.freeze_pretrained(self.freeze)

        return model

    def load(self, ckpt:str) -> transformers.PreTrainedModel:

        # load the model from the checkpoint
        model = self.task.auto_class.from_pretrained(ckpt)

        # create head config
        h_config = self.task.head_config_class(
            head_name=self.head_name,
            label_column=self.label_column
        ) if self.label_column is not None else h.head_config_class(
            head_name=self.head_name
        )
        # update label space information
        h_config.num_labels = model.config.num_labels
        h_config.id2label = model.config.id2label

        # wrap model
        model = hyped.modeling.transformers.HypedTransformerModelWrapper(
            model=model, h_config=h_config
        )
        # freeze/unfreeze pretrained weights
        model.freeze_pretrained(self.freeze)

        return model
