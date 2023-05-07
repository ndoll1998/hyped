from . import (
    processors,
    filters
)

# type hints
AnyProcessorConfig = \
    processors.TokenizerProcessorConfig

AnyFilterConfig = \
    filters.MinSeqLenFilterConfig

# processor mapping
PROCESSOR_MAPPING = {
    processors.TokenizerProcessorConfig: processors.TokenizerProcessor
}

# filter mapping
FILTER_MAPPING = {
    filters.MinSeqLenFilterConfig: filters.MinSeqLenFilter
}

# helper functions
def get_processor_type_from_config(config:processors.DataProcessorConfig) -> type[processors.DataProcessor]:
    # check type
    if not isinstance(config, processors.DataProcessorConfig):
        raise ValueError("Expected instance of `DataProcessorConfig` but got %s" % type(config))
    # get processor from lookup table
    return PROCESSOR_MAPPING[type(config)]

def get_filter_type_from_config(config:filters.DataFilterConfig) -> type[filters.DataFilter]:
    # check type
    if not isinstance(config, filters.DataFilterConfig):
        raise ValueError("Expected instance of `DataFilterConfig` but got %s" % type(config))
    # get filter from lookup table
    return FILTER_MAPPING[type(config)]
