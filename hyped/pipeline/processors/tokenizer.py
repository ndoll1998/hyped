import numpy as np
from .base import DataProcessor, DataProcessorConfig
from transformers import AutoTokenizer
from datasets import Features, Sequence, Value
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Any

@dataclass
class TokenizerProcessorConfig(DataProcessorConfig):
    processor_type:Literal["tokenizer"] = "tokenizer"
    # pretrained tokenizer and column to use
    pretrained_ckpt:str = "bert-base-uncased"
    text_column:str = "text"
    # tokenization arguments
    add_special_tokens:bool =True
    padding:bool | Literal[
        'max_length',
        'do_not_pad'
    ] =False
    truncation:bool | Literal[
            'longest_first',
            'only_first',
            'only_second',
            'do_not_truncate'
    ] =False
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
    return_word_ids:bool =False

class TokenizerProcessor(DataProcessor):
    """Tokenizer Data Processor"""

    def __init__(self, config:DataProcessorConfig) -> None:
        super(TokenizerProcessor, self).__init__(config=config)
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_ckpt,
            use_fast=True,
            add_prefix_space=True
        )

    def map_features(self, features:Features) -> Features:

        # make sure text column is present
        if self.config.text_column not in features:
            raise KeyError("`%s` not present in features!" % self.config.text_column)
        # check type of input feature
        f = features[self.config.text_column]
        if self.config.is_split_into_words and not (isinstance(f, Sequence) and (f.feature == Value('string'))):
            raise TypeError("Input feature `%s` must be sequence of strings, got %s." % (self.config.text_column, features[self.config.text_column]))
        elif (not self.config.is_split_into_words) and (f != Value('string')):
            raise TypeError("Input feature `%s` must be string, got %s." % (self.config.text_column, features[self.config.text_column]))

        # check for constant length
        is_constant = (self.config.max_length is not None) and \
            (self.config.padding == 'max_length') and \
            (self.config.truncation in (True, 'longest_first', 'only_first', 'only_second'))
        length = self.config.max_length if is_constant else -1

        # create new features
        new_features = Features()
        # add features
        new_features['input_ids'] = Sequence(Value(dtype='int64'), length=length)
        if self.config.return_token_type_ids:
            new_features['token_type_ids'] = Sequence(Value(dtype='int64'), length=length)
        if self.config.return_attention_mask:
            new_features['attention_mask'] = Sequence(Value(dtype='int32'), length=length)
        if self.config.return_overflowing_tokens:
            new_features['overflowing_tokens'] = Sequence(Value(dtype='string'))
            new_features['num_truncated_tokens'] = Value(dtype='int32')
        if self.config.return_special_tokens_mask:
            new_features['special_tokens_mask'] = Sequence(Value(dtype='int32'), length=length)
        if self.config.return_special_tokens_mask:
            new_features['special_tokens_mask'] = Sequence(Value(dtype='int32'), length=length)
        if self.config.return_length:
            new_features['length'] = Value(dtype='int32')
        if self.config.return_word_ids:
            new_features['word_ids'] = Sequence(Value(dtype='int32'), length=length)
        # return updated features
        return new_features

    @property
    def tokenization_kwargs(self) -> dict:
        kwargs = asdict(self.config)
        kwargs.pop('processor_type')
        kwargs.pop('pretrained_ckpt')
        kwargs.pop('text_column')
        kwargs.pop('return_word_ids')
        return kwargs

    def process(self, example:dict[str, Any]) -> dict[str, np.ndarray]:
        # apply tokenizer
        enc = self.tokenizer(
            text=example[self.config.text_column],
            **self.tokenization_kwargs
        )
        # add word ids to encoding
        if self.config.return_word_ids:
            enc['word_ids'] = [(i if i is not None else -1) for i in enc.word_ids()]
        # return encoding
        return enc
