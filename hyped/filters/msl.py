from .filter import DataFilter, DataFilterConfig
from dataclasses import dataclass
from typing import Any, Literal
from transformers import PreTrainedTokenizer

@dataclass
class MinSeqLenFilterConfig(DataFilterConfig):
    type:Literal['min-seq-len-filter'] = 'min-seq-len-filter'
    # minimum number of tokens
    min_length:int =16

class MinSeqLenFilter(DataFilter):
    """Minimum Sequence Length Filter

    Filter function filtering out all elements with the
    too few valid tokens (i.e. non-special tokens)
    On top of ignoreing padding tokens, this also includes
    unknown tokens and other spacial tokens specific to the tokenizer
    """

    def __init__(
        self,
        tokenizer:PreTrainedTokenizer,
        config:MinSeqLenFilterConfig
    ) -> None:
        super(MinSeqLenFilter, self).__init__(
            tokenizer=tokenizer,
            config=config
        )

    def filter(self, example:Any) -> bool:
        # count the number of special tokens and check against threshold
        n_special_tokens = sum(self.tokenizer.get_special_tokens_mask(
            example['input_ids'], already_has_special_tokens=True))
        return len(example['input_ids']) - n_special_tokens > self.config.min_length
