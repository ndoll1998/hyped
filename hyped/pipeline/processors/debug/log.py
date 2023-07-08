from ..jinja import JinjaProcessor, JinjaProcessorConfig
from datasets import Features
from dataclasses import dataclass
from typing import Literal, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class LogProcessorConfig(JinjaProcessorConfig):
    processor_type:Literal["debug.log"] = "debug.log"

    # log level to use
    level:int|Literal[
        "DEBUG", "INFO", "ERROR", "WARNING"
    ] = "DEBUG"
    # default template
    template:str = "{{ item }}"
    # fix output column for log
    # note that the column is not added to the features
    output_column:str = "__log__"

    def __post_init__(self):
        super(LogProcessorConfig, self).__post_init__()
        # get log level from name
        if not isinstance(self.level, int):
            self.level = logging.getLevelName(self.level)

class LogProcessor(JinjaProcessor):

    def map_features(self, features:Features) -> Features:
        return features

    def process(self, example:dict[str, Any], index:int, rank:int) -> dict[str, Any]:
        # generate log
        log = super(LogProcessor, self).process(example, index, rank)
        log = log[self.config.output_column]
        # log if there is something to log
        if len(log) > 0:
            logger.log(self.config.level, log)
        # noting to add here
        return {}
