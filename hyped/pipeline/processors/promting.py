from .base import DataProcessor, DataProcessorConfig
from datasets import Features, Value
from jinja2 import Environment as JinjaEnv, StrictUndefined
from dataclasses import dataclass
from typing import Literal, Any

@dataclass
class PromtingProcessorConfig(DataProcessorConfig):
    processor_type:Literal["promting"] = "promting"
    # promt - can be a jinja template
    promt:str =None
    # output
    output_column:str ="promt"

class PromtingProcessor(DataProcessor):
    """Promting Data Processor"""

    def __init__(self, config:PromtingProcessorConfig) -> None:
        super(PromtingProcessor, self).__init__(config=config)
        # create jinja template
        self.jinja_env = JinjaEnv(undefined=StrictUndefined)
        # TODO: extract variables from template and check for
        self.promt_template = self.jinja_env.from_string(self.config.promt)

    def map_features(self, features:Features) -> Features:
        return Features({self.config.output_column: Value(dtype="string")})

    def process(self, example:dict[str, Any]) -> dict[str, Any]:
        return {
            self.config.output_column: self.promt_template.render(
                item=example, features=self.in_features
            )
        }
