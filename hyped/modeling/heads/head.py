import torch
import datasets
import transformers
from transformers.modeling_outputs import ModelOutput
from abc import ABC, ABCMeta, abstractmethod

class PredictionHeadOutput(transformers.modeling_outputs.ModelOutput):
    loss:None|torch.FloatTensor =None
    logits:None|torch.FloatTensor =None

class PredictionHeadConfig(transformers.PretrainedConfig, ABC):
    model_type = None

    def __init__(
        self,
        label_column:str ="labels",
        loss_coeff:float =1.0,
        **kwargs
    ) -> None:
        super(PredictionHeadConfig, self).__init__(**kwargs)
        # save label column
        self.label_column = label_column
        self.loss_coeff = loss_coeff

    def check_and_prepare(self, features:datasets.Features) -> list[str]:
        # get label space and set attributes accordingly
        labels = self.get_label_space(features)
        self.num_labels = len(labels)
        self.id2label = dict(enumerate(labels))
        self.label2id = {label:i for i, label in self.id2label.items()}
        # also return the label space because why not
        return labels

    @abstractmethod
    def get_label_space(self, features:datasets.Features) -> list[str]:
        ...

class RegisterHeadMeta(ABCMeta):

    def __new__(cls, name, bases, namespaces):
        # create type
        t = ABCMeta.__new__(cls, name, bases, namespaces)
        # register config and head to auto classes if
        # config class is set correctly
        if issubclass(t.config_class, PredictionHeadConfig) and (t.config_class.model_type is not None):
            transformers.AutoConfig.register(t.config_class.model_type, t.config_class)
            transformers.AutoModel.register(t.config_class, t)
        # return new type
        return t

class PredictionHead(transformers.PreTrainedModel, ABC, metaclass=RegisterHeadMeta):

    config_class = PredictionHeadConfig
    _keys_to_ignore_on_load_missing = []

    def __init__(
        self,
        config:PredictionHeadConfig,
        encoder_config:transformers.PretrainedConfig
    ) -> None:
        super(PredictionHead, self).__init__(config)

    @abstractmethod
    def forward(self, encoder_output:ModelOutput, labels:None|torch.Tensor =None) -> PredictionHeadOutput:
        ...
