from .base import DataProcessor, DataProcessorConfig
from datasets import Features, Sequence, Value
from dataclasses import dataclass
from typing import Literal, Any, Iterator

@dataclass
class ChunkProcessorConfig(DataProcessorConfig):
    processor_type:Literal["chunk"] = "chunk"
    
    stride:int = 0
    chunksize:int = None
    columns:list[str] = None

    def __post_init__(self) -> None:
        # check arguments
        if self.chunksize is None:
            raise ValueError("No chunksize specified")
        if self.columns is None:
            raise ValueError("No columns specified")

class ChunkProcessor(DataProcessor):
    """Chunk Data Processor"""

    def map_features(self, features:Features) -> Features:

        for col in self.config.columns:
            if col not in features:
                raise ValueError("Column `%s` not present in dataset features!" % col)
            if not isinstance(features[col], Sequence):
                raise TypeError("Column `%s` is not a sequence, thus cannot be chunked!" % col)

        # chunk meta information
        features['chunk_id'] = Value(dtype='int32')
        features['chunk_source_id'] = Value(dtype='int32')

        # TODO: check sequence lengths
        return features

    def chunk(self, example:dict[str, list[Any]]) -> Iterator[dict[str, list[Any]]]:
        
        l = len(next(iter(example.values())))
        for offset in range(0, l, self.config.chunksize - self.config.stride):

            yield {
                k: v[offset:offset + self.config.chunksize]
                for k, v in example.items()
            }

    def process(self, examples:dict[str, list[Any]], index:list[int]) -> dict[str, Any]:

        chunked_examples = {k: [] for k in examples.keys()} | {
            'chunk_id': [],
            'chunk_source_id': []
        }
        # get the batch size
        n = len(next(iter(examples.values())))
        # apply processor to each item in the batch seperately
        for i, idx in enumerate(index):
            # extract single example from batch
            example = {k: examples[k][i] for k in self.config.columns}

            # iterate over all chunks
            for j, chunk in enumerate(self.chunk(example)):
                # meta information
                chunked_examples['chunk_id'].append(j)
                chunked_examples['chunk_source_id'].append(idx)
                # collect all chunks
                for k, v in chunk.items():
                    chunked_examples[k].append(v)
                # create copies for all other features
                for k, v in examples.items():
                    if k not in self.config.columns:
                        chunked_examples[k].append(v[i])

        return chunked_examples
