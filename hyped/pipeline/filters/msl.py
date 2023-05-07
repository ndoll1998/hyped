from .base import DataFilter, DataFilterConfig
from dataclasses import dataclass
from typing import Any, Literal
from transformers import AutoTokenizer

@dataclass
class MinSeqLenFilterConfig(DataFilterConfig):
    filter_type:Literal['min-seq-len-filter'] = 'min-seq-len-filter'
    # minimum number of tokens
    pretrained_ckpt:str ="bert-base-uncased"
    min_length:int =16

class MinSeqLenFilter(DataFilter):
    """Minimum Sequence Length Filter

    Filter function filtering out all elements with the
    too few valid tokens (i.e. non-special tokens)
    On top of ignoreing padding tokens, this also includes
    unknown tokens and other spacial tokens specific to the tokenizer
    """

    def __init__(self, config:MinSeqLenFilterConfig) -> None:
        super(MinSeqLenFilter, self).__init__(config)
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_ckpt)

    def filter(self, example:Any) -> bool:
        # count the number of special tokens and check against threshold
        n_special_tokens = sum(self.tokenizer.get_special_tokens_mask(
            example['input_ids'], already_has_special_tokens=True))
        return len(example['input_ids']) - n_special_tokens > self.config.min_length
