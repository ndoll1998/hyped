from ..wrapper import HypedModelWrapper
from ..heads import HypedHeadConfig
from transformers import PreTrainedModel
from transformers.utils.generic import find_labels

class HypedTransformerModelWrapper(HypedModelWrapper):

    __slots__ = ('__h_config__', '__label_name__')

    def __init__(
        self,
        model:PreTrainedModel,
        h_config:HypedHeadConfig
    ) -> None:
        # only supports single label column tasks
        if len(h_config.label_columns) != 1:
            raise NotImplementedError()
        # get label names expected by model
        label_names = find_labels(type(model)) or ['labels']
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
