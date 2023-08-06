import transformers
from ..wrapper import HypedModelWrapper
from ..heads import HypedHeadConfig
from .utils import get_pretrained_module

class HypedTransformerModelWrapper(HypedModelWrapper):

    __slots__ = ('__h_config__', '__label_name__')

    def __init__(
        self,
        model:transformers.PreTrainedModel,
        h_config:HypedHeadConfig
    ) -> None:
        # only supports single label column tasks
        if len(h_config.label_columns) != 1:
            raise NotImplementedError()
        # get label names expected by model
        label_names = transformers.utils.generic.find_labels(type(model)) or ['labels']
        assert len(label_names) == len(h_config.label_columns)

        # save head config and label name expected by model
        self.__h_config__ = h_config
        self.__label_name__ = label_names[0]
        # initialize wrapper
        super(HypedTransformerModelWrapper, self).__init__(model)

    def __call__(self, *args, **kwargs):
        # rename label column
        kwargs = kwargs.copy()
        kwargs[self.__label_name__] = kwargs.get(self.__h_config__.label_columns[0])
        # apply model and return output
        return super(HypedTransformerModelWrapper, self).__call__(*args, **kwargs)

    @property
    def head_configs(self) -> list[HypedHeadConfig]:
        return [self.__h_config__]

    def freeze_pretrained(self, freeze:bool =True) -> None:
        # get module of pretrained weights and freeze/unfreeze it's parameters
        for p in get_pretrained_module(self).parameters():
            p.requires_grad_(not freeze)

