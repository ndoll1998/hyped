import datasets
from .auto import (
    AutoDataProcessor,
    AutoDataFilter
)
from .processors.base import DataProcessor, DataProcessorConfig
from .filters.base import DataFilter, DataFilterConfig
# utils
from typing import TypeVar
from hyped.utils.typedlist import typedlist

T = TypeVar("T")

class DataProcessorList(typedlist[T]):

    def handle_type_conflict(self, config:DataProcessorConfig) -> T:
        # try to create a processor instance from the config
        return AutoDataProcessor.from_config(config)

class DataFilterList(typedlist[T]):

    def handle_type_conflict(self, config:DataFilterConfig) -> T:
        # try to create a processor instance from the config
        return AutoDataFilter.from_config(config)

class Pipeline(object):

    def __init__(
        self,
        processors:list[DataProcessor|DataProcessorConfig] =[],
        filters:list[DataFilter|DataFilterConfig] =[]
    ) -> None:
        # initialize processor and filter lists
        self.processors = DataProcessorList[DataProcessor]()
        self.filters = DataFilterList[DataFilter]()
        # add processors and filters
        self.processors.extend(processors)
        self.filters.extend(filters)

    @property
    def in_features(self) -> datasets.Features:
        return self.processors[0].in_features

    @property
    def out_features(self) -> datasets.Features:
        return self.processors[-1].out_features

    def prepare(self, features:datasets.Features) -> datasets.Features:
        # prepare all processors
        for p in self.processors:
            assert isinstance(p, DataProcessor)
            features = p.prepare(features)
        # prepare pipeline
        return features

    def __call__(self,
        ds:datasets.Dataset|datasets.DatasetDict,
        use_cache:bool =False
    ) -> datasets.Dataset|datasets.DatasetDict:
        # check input type
        if not isinstance(ds, (datasets.Dataset, datasets.DatasetDict)):
            raise ValueError("Expected `ds` to be a `datasets.Dataset` or `datasets.DatasetDict`, got %s" % type(ds))

        # apply processors
        for p in self.processors:
            assert isinstance(p, DataProcessor)
            ds = ds.map(
                function=p,
                with_indices=p.requires_index,
                with_rank=p.requires_rank,
                batched=False, # TODO: support batched processing
                load_from_cache_file=use_cache,
                desc=type(p).__name__
            )

        # apply filters
        for f in self.filters:
            assert isinstance(f, DataFilter)
            ds = ds.filter(
                function=f,
                with_indices=f.requires_index,
                batched=False, # TODO
                load_from_cache_file=use_cache,
                desc=type(f).__name__
            )

        # return processes dataset
        return ds
