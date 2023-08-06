from .base import DataProcessor, DataProcessorConfig
from .tokenizer import TokenizerProcessor, TokenizerProcessorConfig
from .bio import BioLabelProcessor, BioLabelProcessorConfig
from .jinja import JinjaProcessorConfig, JinjaProcessor
from .math import MathProcessorConfig, MathProcessor
from .chunk import ChunkProcessorConfig, ChunkProcessor
# debug processors
from .debug.log import LogProcessorConfig, LogProcessor

AnyProcessorConfig = \
    LogProcessorConfig | \
    TokenizerProcessorConfig | \
    BioLabelProcessorConfig | \
    JinjaProcessorConfig | \
    MathProcessorConfig | \
    ChunkProcessorConfig
