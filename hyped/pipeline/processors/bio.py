import logging
import numpy as np
from .base import DataProcessor, DataProcessorConfig
from datasets import Features, Sequence, ClassLabel, Value
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Any

logger = logging.getLogger(__name__)

@dataclass
class BioLabelProcessorConfig(DataProcessorConfig):
    processor_type:Literal["bio-labels"] = "bio-labels"

    word_ids_column:str = "word_ids"
    output_column:str = "bio"
    # label source on token-level
    # TODO: rename to word instead of token to avoid confusion
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

    @property
    def entity_names(self) -> list[str]:
        if self.config.token_bio_column is not None:
            raise NotImplementedError()
        # return entity names from input features
        return self.in_features[self.config.token_span_column]['type'].feature.names

    @property
    def out_tag_id(self) -> int:
        return self.bio_label2id(self.config.out_tag)

    @property
    def bio_tags(self) -> list[str]:
        return self.out_features[self.config.output_column].feature.names

    def bio_label2id(self, values:str|list[str]) -> int|list[int]:
        return self.out_features[self.config.output_column].feature.str2int(values)

    @cached_property
    def begin2in(self) -> np.ndarray:
        # shorthands for begin and in tag prefix
        b_prefix = self.config.begin_tag_prefix
        i_prefix = self.config.in_tag_prefix
        # separate begin and in bio tags
        begin_tags = {tag: tag[len(b_prefix):] for tag in self.bio_tags if tag.startswith(b_prefix)}
        in_tags = {tag: tag[len(i_prefix):] for tag in self.bio_tags if tag.startswith(i_prefix)}

        # make sure there is a begin tag for each in tag
        assert set(begin_tags.values()) == set(in_tags.values())
        # make sure corresponding begin- and in-tags reference the same entity
        assert all(entity == in_tags[tag] for tag, entity in in_tags.items())

        # map begin to corresponding in tag
        begin2in = {tag: i_prefix + entity for tag, entity in begin_tags.items()}
        begin2in = [begin2in.get(tag, tag) for tag in self.bio_tags]
        begin2in = self.bio_label2id(begin2in)
        # convert to tensor
        # this tensor maps the label-id of begin-tags to the label-id of the
        # corresponding in-tags. Label-ids of non-begin-tags remain untouched.
        # Examples:
        #    - begin2in[label2id["B-ORG"]] = label2id["I-ORG"]
        #    - begin2in[label2id["I-ORG"]] = label2id["I-ORG"]
        return np.asarray(begin2in)

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

        out_feature = None
        if self.config.token_bio_column is not None:
            # check if token bio labels column is present
            if self.config.token_bio_column not in features:
                raise KeyError("`%s` not present in features!" % self.config.token_bio_column)
            # check type of bio labels feature
            f = features[self.config.token_bio_column]
            if not (isinstance(f, Sequence) and isinstance(f.feature, ClassLabel)):
                raise TypeError("Expected bio labels to be a `Sequence` of `ClassLabels`, got %s." % f)

            # add feature
            out_feature = Sequence(ClassLabel(names=f.feature.names), length=l)

        elif self.config.token_span_column is not None:
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
            out_feature = Sequence(ClassLabel(names=bio_tags), length=l)

        assert out_feature is not None
        return Features({self.config.output_column: out_feature})

    def process(self, example:dict[str, Any]) -> dict[str, np.ndarray]:

        # get word ids from examples and compute special tokens mask
        word_ids = np.asarray(example[self.config.word_ids_column])
        special_tokens_mask = (word_ids < 0)

        if self.config.token_bio_column is not None:
            # get token-level bio scheme and map it to word level
            bio = np.asarray(example[self.config.token_bio_column])
            bio = np.where(special_tokens_mask, self.config.ignore_label_index, bio[word_ids])
            # mask all tags that should be in-tags but are begin-tags
            in_mask = np.zeros_like(bio, dtype=bool)
            in_mask[1:] = (word_ids[:-1] == word_ids[1:])
            in_mask &= ~special_tokens_mask
            # convert all begin tags that should be in tags
            bio[in_mask] = self.begin2in[bio[in_mask]]

        elif self.config.token_span_column is not None:
            # build initial empty bio labels and get spans
            bio = np.where(special_tokens_mask, self.config.ignore_label_index, self.out_tag_id)
            spans = example[self.config.token_span_column]

            # process each span
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
                entity = self.entity_names[entity_t]
                bio[idx[0]] = self.bio_label2id(self.config.begin_tag_prefix + entity)
                bio[idx[1:]] = self.bio_label2id(self.config.in_tag_prefix + entity)

        # add bio to example
        example[self.config.output_column] = bio
        return example
