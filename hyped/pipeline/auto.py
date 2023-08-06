from . import processors
from .processors.base import DataProcessor, DataProcessorConfig
# utils
from hyped.utils.typedmapping import typedmapping

class AutoDataProcessor(object):
    MAPPING = typedmapping[
        type[DataProcessorConfig],
        type[DataProcessor]
    ]()

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

# register all processors
AutoDataProcessor.register(processors.LogProcessorConfig, processors.LogProcessor)
AutoDataProcessor.register(processors.TokenizerProcessorConfig, processors.TokenizerProcessor)
AutoDataProcessor.register(processors.BioLabelProcessorConfig, processors.BioLabelProcessor)
AutoDataProcessor.register(processors.JinjaProcessorConfig, processors.JinjaProcessor)
AutoDataProcessor.register(processors.MathProcessorConfig, processors.MathProcessor)
AutoDataProcessor.register(processors.ChunkProcessorConfig, processors.ChunkProcessor)
