from transformers import AutoTokenizer
from hyped.data.processors.tokenizer import (
    TokenizerProcessorConfig,
    TokenizerProcessor,
)


class TestTokenizerProcessor:
    def test_tokenization(self):
        example = "This is a test sentence."
        # create tokenizer processor and tokenizer instance
        p = TokenizerProcessor(TokenizerProcessorConfig())
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # apply processor
        out = p.process(example={"text": example}, index=0, rank=0)
        # compare outputs
        assert out["input_ids"] == tokenizer(example).input_ids
