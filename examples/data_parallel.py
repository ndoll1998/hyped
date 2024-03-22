from dataclasses import dataclass
from typing import Any

import datasets
import torch
import transformers

from hyped.data.io.writers.json import JsonDatasetWriter
from hyped.data.pipe import DataPipe
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.data.processors.tokenizers.hf import (
    HuggingFaceTokenizer,
    HuggingFaceTokenizerConfig,
)
from hyped.utils.feature_access import FeatureKey, batch_get_value_at_key


@dataclass
class ApplyModelProcessorConfig(BaseDataProcessorConfig):
    pretrained_ckpt: str = None

    input_ids: FeatureKey = "input_ids"
    attention_mask: FeatureKey = "attention_mask"


class ApplyModelProcessor(BaseDataProcessor[ApplyModelProcessorConfig]):
    def __init__(self, config: ApplyModelProcessorConfig) -> None:
        super(ApplyModelProcessor, self).__init__(config)
        # device and model placeholders
        self.device: torch.device = None
        self.model: transformers.PreTrainedModel = None

    def map_features(self, features: datasets.Features) -> datasets.Features:
        return {}

    @torch.no_grad()
    def internal_batch_process(
        self, examples: dict[str, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[str, list[Any]], list[int]]:
        # load model
        if self.model is None:
            self.device = torch.device("cuda:%i" % rank)
            self.model = transformers.AutoModel.from_pretrained(
                self.config.pretrained_ckpt
            ).to(self.device)

        # get batch of input ids
        input_ids = batch_get_value_at_key(examples, self.config.input_ids)
        attention_mask = batch_get_value_at_key(
            examples, self.config.attention_mask
        )
        # convert to pytorch tensors and move to device
        input_ids = torch.LongTensor(input_ids).to(self.device)
        attention_mask = torch.BoolTensor(attention_mask).to(self.device)
        # apply model
        self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # post-process model outputs
        ...

        return {}, range(len(index))


if __name__ == "__main__":
    # load dataset
    ds = datasets.load_dataset("imdb", split="train")  # , streaming=True)
    ds = ds.to_iterable_dataset(num_shards=8)

    # define data pipeline
    pipe = DataPipe(
        [
            HuggingFaceTokenizer(
                HuggingFaceTokenizerConfig(
                    tokenizer="bert-base-uncased",
                    text="text",
                    max_length=128,
                    padding="max_length",
                    truncation=True,
                    return_attention_mask=True,
                )
            ),
            ApplyModelProcessor(
                ApplyModelProcessorConfig(pretrained_ckpt="bert-base-uncased")
            ),
        ]
    )

    # apply dataset and write to output
    ds = pipe.apply(ds)
    JsonDatasetWriter(
        save_dir="output",
        exist_ok=True,
        # use one process per gpu
        num_proc=torch.cuda.device_count(),
    ).consume(ds)