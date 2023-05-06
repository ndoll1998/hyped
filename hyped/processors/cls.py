from .processor import DataProcessor, DataProcessorConfig
from dataclasses import dataclass
from typing import Any, Literal

from transformers import PreTrainedTokenizer
from datasets.tasks import TextClassification

@dataclass
class TextClsProcessorConfig(DataProcessorConfig):
    """Configuration for Text Classification Data Processor"""
    # processor type
    type:Literal['text-cls-processor'] = 'text-cls-processor'
    # dataset columns of interest
    text_column:str ="text"
    label_column:str ="label"
    # preprocessing parameters
    max_length:int =256

class TextClsProcessor(DataProcessor):
    """Data Processor for Text Classification Tasks"""

    def __init__(
        self,
        tokenizer:PreTrainedTokenizer,
        config:TextClsProcessorConfig
    ) -> None:
        super(TextClsProcessor, self).__init__(
            tokenizer=tokenizer,
            config=config
        )

    @property
    def template(self) -> TextClassification:
        return TextClassification(
            text_column=self.config.text_column,
            label_column=self.config.label_column
        )

    def process(self, example:dict[str, Any]) -> dict[str, Any]:
        return self.tokenizer(
            text=example['text'],
            max_length=self.config.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False
        ) | {'labels': example['labels']}
