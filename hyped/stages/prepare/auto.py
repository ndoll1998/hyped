from . import filters
from . import processors
# base classes
from .filters.base import DataFilter, DataFilterConfig
from .processors.base import DataProcessor, DataProcessorConfig
# utils
from hyped.utils.typedmapping import typedmapping

class AutoClass(object):
    MAPPING:typedmapping

    def __init__(self):
        raise EnvironmentError("AutoClasses are designed to be instantiated using the `AutoClass.from_config(config)` method.")

    @classmethod
    def from_config(cls, config, **kwargs):
        # check if config is present in mapping
        if type(config) not in cls.MAPPING:
            raise KeyError(type(config))
        # create processor instance
        processor_t = cls.MAPPING[type(config)]
        return processor_t(config, **kwargs)

    @classmethod
    def register(cls, config_t, processor_t):
        cls.MAPPING[config_t] = processor_t

class AutoDataProcessor(AutoClass):
    MAPPING = typedmapping[
        type[DataProcessorConfig],
        type[DataProcessor]
    ]()

class AutoDataFilter(AutoClass):
    MAPPING = typedmapping[
        type[DataFilterConfig],
        type[DataFilter]
    ]()

# register all processors
AutoDataProcessor.register(processors.TokenizerProcessorConfig, processors.TokenizerProcessor)
AutoDataProcessor.register(processors.BioLabelProcessorConfig, processors.BioLabelProcessor)
# register all filters
AutoDataFilter.register(filters.MinSeqLenFilterConfig, filters.MinSeqLenFilter)
