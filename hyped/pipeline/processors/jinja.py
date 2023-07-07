from .base import DataProcessor, DataProcessorConfig
from datasets import Features, Value
from jinja2 import Environment as JinjaEnv, StrictUndefined
from dataclasses import dataclass
from typing import Literal, Any

@dataclass
class JinjaProcessorConfig(DataProcessorConfig):
    processor_type:Literal["jinja"] = "jinja"
    # jinja template and output column
    template:str = None
    output_column:str = None

    def __post_init__(self) -> None:
        # check arguments
        if self.template is None:
            raise ValueError("Template not defined")
        if self.output_column is None:
            raise ValueError("Output column not defined, please specify `output_column`")

class JinjaProcessor(DataProcessor):
    """Jinja Data Processor"""

    def __init__(self, config:JinjaProcessorConfig) -> None:
        super(JinjaProcessor, self).__init__(config=config)
        # create jinja template
        self.jinja_env = JinjaEnv(undefined=StrictUndefined)
        # TODO: extract variables from template and check them
        self.template = self.jinja_env.from_string(self.config.template)

    def map_features(self, features:Features) -> Features:
        return Features({self.config.output_column: Value(dtype="string")})

    def process(self, example:dict[str, Any]) -> dict[str, Any]:
        return {
            self.config.output_column: self.template.render(
                item=example, features=self.in_features
            )
        }
