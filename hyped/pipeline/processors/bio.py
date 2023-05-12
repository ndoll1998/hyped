import logging
import numpy as np
from .base import DataProcessor, DataProcessorConfig
from datasets import Features, Sequence, ClassLabel, Value
from dataclasses import dataclass
from typing import Literal, Any

logger = logging.getLogger(__name__)

@dataclass
class BioLabelProcessorConfig(DataProcessorConfig):
    processor_type:Literal["bio-labels"] = "bio-labels"

    word_ids_column:str = "word_ids"
    output_column:str = "bio"
    # 
    token_bio_column:None|str = None
    token_span_column:None|str = None
    # bio scheme
    out_tag:str = "O"
    begin_tag_prefix:str = "B-"
    in_tag_prefix:str = "I-"

    # label index to mark ignore
    ignore_label_index:int =-100

    def __post_init__(self):
        if (self.token_bio_column is None) and (self.token_span_column is None):
            raise ValueError("Either `token_bio_column` or `token_span_column` must be provided, got None.")
        if (self.token_bio_column is not None) and (self.token_span_column is not None):
            raise ValueError("Either `token_bio_column` or `token_span_column` must be provided, got both.")

class BioLabelProcessor(DataProcessor):
    """BIO Labeling Scheme Processor"""

    def map_features(self, features:Features) -> Features:

        # make sure word ids are present in features
        if self.config.word_ids_column not in features:
            raise KeyError("`%s` not present in features!" % self.config.word_ids_column)
        # check type of word ids column
        f = features[self.config.word_ids_column]
        if not (isinstance(f, Sequence) and (f.feature == Value('int32'))):
            raise TypeError("Expected word ids to be a sequence of ints, got %s." % feature)

        # get length of word-ids sequence
        l = f.length

        if self.config.token_bio_column:
            raise NotImplementedError()

        elif self.config.token_span_column:
            # check if token span column is present
            if self.config.token_span_column not in features:
                raise KeyError("`%s` not present in features!" % self.config.token_span_column)
            # check type of span column
            # TODO: check feature type (must containt begin, end, type of correct feature types)
            f = features[self.config.token_span_column]

            # build bio tags
            names = f['type'].feature.names
            bio_tags = [self.config.out_tag] + [
                "%s%s" % (prefix, name) for name in names for prefix in (
                    self.config.begin_tag_prefix,
                    self.config.in_tag_prefix
                )
            ]
            # add feature
            features[self.config.output_column] = Sequence(ClassLabel(names=bio_tags), length=l)

        # return updated features
        assert self.config.output_column in features
        return features

    def process(self, example:dict[str, Any]) -> dict[str, np.ndarray]:

        # get word ids from examples and create initial bio labels
        word_ids = np.asarray(example[self.config.word_ids_column])
        bio = np.where(word_ids < 0, self.config.ignore_label_index, 0)

        spans = example[self.config.token_span_column]
        for entity_t, begin, end in zip(spans['type'], spans['begin'], spans['end']):
            # get entity mask
            mask = (begin <= word_ids) & (word_ids < end)
            # any tokens, mostly occurs for out of bounds entities
            if not mask.any():
                logger.warning("Detected entity out of bounds, skipping entity.")
                continue
            # handle entity overlaps
            if (bio[mask] != 0).any():
                logger.warning("Detected entity overlap, skipping entity.")
                continue
            # update bio labels
            idx, = mask.nonzero()
            bio[idx[0]] = 1 + 2 * entity_t
            bio[idx[1:]] = 1 + 2 * entity_t + 1

        # add bio to example
        example['bio'] = bio
        return example
