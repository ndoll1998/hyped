from .base import DataProcessor, DataProcessorConfig
from .tokenizer import TokenizerProcessor, TokenizerProcessorConfig
from .bio import BioLabelProcessor, BioLabelProcessorConfig
from .jinja import JinjaProcessorConfig, JinjaProcessor

AnyProcessorConfig = \
    TokenizerProcessorConfig | \
    BioLabelProcessorConfig | \
    JinjaProcessorConfig
