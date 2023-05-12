from . import filters
from . import processors
# base classes
from .filters.base import DataFilter, DataFilterConfig
from .processors.base import DataProcessor, DataProcessorConfig
# utils
from typing import Generic, TypeVar, Any
from hyped.utils.typedmapping import typedmapping

C = TypeVar('C')
T = TypeVar('T')

class ConfigMapping(typedmapping[C, T]):

    def check_key_type(self, key:Any) -> C:
        # handle type conflict if value has incorrect type
        if not isinstance(key, type):
            raise TypeError("Excepted key to be a type object, got %s." % key)
        if not issubclass(key, self._K):
            raise TypeError("Expected key to be sub-type of %s, got %s." % (self._K, key))
        # otherwise all fine
        return key

    def check_val_type(self, val:Any) -> T:
        # handle type conflict if value has incorrect type
        if not isinstance(val, type):
            raise TypeError("Excepted key to be a type object, got %s." % val)
        if not issubclass(val, self._V):
            raise TypeError("Expected value to be sub-type of %s, got %s." % (self._V, val))
        # otherwise all fine
        return val

class AutoClass(object):
    MAPPING:ConfigMapping

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
    MAPPING = ConfigMapping[DataProcessorConfig, DataProcessor]()

class AutoDataFilter(AutoClass):
    MAPPING = ConfigMapping[DataFilterConfig, DataFilter]()

# register all processors
AutoDataProcessor.register(processors.TokenizerProcessorConfig, processors.TokenizerProcessor)
AutoDataProcessor.register(processors.BioLabelProcessorConfig, processors.BioLabelProcessor)
# register all filters
AutoDataFilter.register(filters.MinSeqLenFilterConfig, filters.MinSeqLenFilter)
