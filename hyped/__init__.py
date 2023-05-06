from typing import Union
from . import (
    processors,
    filters
)

__all__ = [
    'processors',
    'filters'
]

# type hints
AnyProcessorConfig = Union[
    processors.TextClsProcessorConfig,
    processors.BioTaggingProcessorConfig,
]
AnyFilterConfig = Union[
    filters.MinSeqLenFilterConfig
]

# processor mapping
PROCESSOR_MAPPING = {
    processors.TextClsProcessorConfig: processors.TextClsProcessor,
    processors.BioTaggingProcessorConfig: processors.BioTaggingProcessor
}

# filter mapping
FILTER_MAPPING = {
    filters.MinSeqLenFilterConfig: filters.MinSeqLenFilter
}

# helper functions
def get_processor_type_from_config(config:processors.DataProcessorConfig) -> type[processors.DataProcessor]:
    return PROCESSOR_MAPPING[type(config)]

def get_filter_type_from_config(config:filters.DataFilterConfig) -> type[filters.DataFilter]:
    return FILTER_MAPPING[type(config)]

