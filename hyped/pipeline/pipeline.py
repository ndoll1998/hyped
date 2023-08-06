import os
import datasets
import pyarrow as pa
from .auto import AutoDataProcessor
from .processors.base import (
    DataProcessor,
    DataProcessorConfig
)
# utils
from hyped.utils.typedlist import typedlist
from typing import Any

class Pipeline(DataProcessor, typedlist[DataProcessor]):

    def handle_type_conflict(self, config:DataProcessorConfig) -> DataProcessor:
        # try to create a processor instance from the config
        return AutoDataProcessor.from_config(config)

    def __init__(
        self,
        processors:list[DataProcessor|DataProcessorConfig] =[],
    ) -> None:
        DataProcessor.__init__(self, None)
        # initialize processor list and add all processors
        typedlist.__init__(self)
        self.extend(processors)

    @property
    def config(self) -> list[DataProcessorConfig]:
        return [p.config for p in self]

    @property
    def in_features(self) -> datasets.Features:
        return self[0].in_features

    @property
    def new_features(self) -> datasets.Features:
        return self[-1].new_features

    @property
    def out_features(self) -> datasets.Features:
        return self[-1].out_features

    def map_features(self, features:datasets.Features) -> datasets.Features:
        # map features is unused
        for p in self:
            features = p.map_features(features)
        return features

    def prepare(self, features:datasets.Features) -> datasets.Features:
        # prepare all processors
        for p in self:
            features = p.prepare(features)
        return features

    def process(
        self, examples:dict[str, list[Any]], index:list[int], rank:int
    ) -> dict[str, list[Any]]:
        # apply each processor in pipeline
        for p in self:
            kwargs = (
                ({'index': index} if p.requires_index else {}) |
                ({'rank': rank} if p.requires_rank else {})
            )
            examples = p(examples, **kwargs)

            n = len(next(iter(examples.values())))
            # re-index if needed
            if self.requires_index and (n != len(index)):
                index = list(range(n))

        return examples

    def __call__(self, examples:dict[str, list[Any]], index:list[int], rank:int):
        # process examples
        processed_examples = self.process(examples, index, rank)
        # convert to py-arrow table with correct schema
        return pa.table(
            data=dict(processed_examples),
            schema=self.out_features.arrow_schema
        )

    def apply(self,
        ds:datasets.Dataset|datasets.DatasetDict,
        batch_size:int = 512,
        num_proc:None|int = None,
        use_cache:bool = False,
        desc:None|str = "Preprocess"
    ) -> datasets.Dataset|datasets.DatasetDict:
        # check input type
        if not isinstance(ds, (datasets.Dataset, datasets.DatasetDict)):
            raise ValueError("Expected `ds` to be a `datasets.Dataset` or `datasets.DatasetDict`, got %s" % type(ds))

        # apply pipeline
        return ds.map(
            function=self,
            with_indices=self.requires_index,
            with_rank=self.requires_rank,
            batched=True,
            batch_size=batch_size,
            # num_proc=num_proc or os.cpu_count(),
            load_from_cache_file=use_cache,
            desc=desc
        )
