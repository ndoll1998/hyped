import numpy as np
from .processor import DataProcessor, DataProcessorConfig
from transformers import AutoTokenizer
from dataclasses import dataclass, asdict
from typing import Literal, Union, Optional, Any

@dataclass
class TokenizerProcessorConfig(DataProcessorConfig):
    processor_type:Literal["tokenizer"] = "tokenizer"
    # pretrained tokenizer and column to use
    pretrained_ckpt:str = "bert-base-uncased"
    text_column:str = "text"
    # tokenization arguments
    add_special_tokens:bool =True
    padding:Union[bool, Literal[
        'max_length',
        'do_not_pad'
    ]] =False
    truncation:Union[bool, Literal[
            'longest_first',
            'only_first',
            'only_second',
            'do_not_truncate'
    ]] =False
    max_length:Optional[int] =None
    stride:int =0
    is_split_into_words:bool =False
    pad_to_multiple_of:Optional[int] =None
    # return values
    return_token_type_ids:bool =True
    return_attention_mask:bool =True
    return_overflowing_tokens:bool =False
    return_special_tokens_mask:bool =False
    return_offsets_mapping:bool =False
    return_length:bool =False

class TokenizerProcessor(DataProcessor):
    """Tokenizer Data Processor"""

    def __init__(self, config:DataProcessorConfig) -> None:
        super(TokenizerProcessor, self).__init__(config=config)
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_ckpt,
            use_fast=True
        )

    @property
    def tokenization_kwargs(self) -> dict:
        kwargs = asdict(self.config)
        kwargs.pop('processor_type')
        kwargs.pop('pretrained_ckpt')
        kwargs.pop('text_column')
        return kwargs

    def process(self, example:dict[str, Any]) -> dict[str, np.ndarray]:
        return self.tokenizer(
            text=example['text'],
            **self.tokenization_kwargs
        )
