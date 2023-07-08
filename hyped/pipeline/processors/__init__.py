from .base import DataProcessor, DataProcessorConfig
from .tokenizer import TokenizerProcessor, TokenizerProcessorConfig
from .bio import BioLabelProcessor, BioLabelProcessorConfig
from .jinja import JinjaProcessorConfig, JinjaProcessor
from .math import MathProcessorConfig, MathProcessor
# debug processors
from .debug.log import LogProcessorConfig, LogProcessor

AnyProcessorConfig = \
    LogProcessorConfig | \
    TokenizerProcessorConfig | \
    BioLabelProcessorConfig | \
    JinjaProcessorConfig | \
    MathProcessorConfig
