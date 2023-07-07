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
    char_offsets_column:str = "offset_mapping"
    # label source on word- or character-level
    word_bio_column:None|str = None
    word_span_column:None|str = None
    char_span_column:None|str = None

    # bio scheme
    out_tag:str = "O"
    begin_tag_prefix:str = "B-"
    in_tag_prefix:str = "I-"
    # dataset column to store generated bio scheme
    output_column:str = "bio"

    # label index to mark ignore
    ignore_label_index:int =-100

    def __post_init__(self):

        sources = {
            'word_bio_column': self.word_bio_column,
            'word_span_column': self.word_span_column,
            'char_span_column': self.char_span_column
        }
        sources = {k: v for k, v in sources.items() if v is not None}

        if len(sources) == 0:
            raise ValueError(
                "Either `word_bio_column`, `word_span_column` or `char_span_column` must be provided, got %s." % str(sources)
            )
        if len(sources) > 1:
            raise ValueError("Multiple BIO label sources specified, got `%s`" % str(sources))

        if (self.word_ids_column is None) and (
            (self.word_bio_column is not None) or
            (self.word_span_column is not None)
        ):
            raise ValueError("`word_ids_column` required for `%s`" % str(sources))

        if (self.char_offsets_column is None) and (self.char_span_column is not None):
            raise ValueError("`char_offsets_column` required for `%s`" % str(sources))


class FromWordLevelLabels(DataProcessor):
    """Bio label processor backbone for generating token-level labels from word-level labels"""

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

    def map_features(self, features:Features) -> Sequence:

        # make sure word ids are present in features
        if self.config.word_ids_column not in features:
            raise KeyError("`%s` not present in features!" % self.config.word_ids_column)
        # check type of word ids column
        f = features[self.config.word_ids_column]
        if not (isinstance(f, Sequence) and (f.feature == Value('int32'))):
            raise TypeError("Expected word ids to be a sequence of ints, got %s." % f)

        # check if token bio labels column is present
        if self.config.word_bio_column not in features:
            raise KeyError("`%s` not present in features!" % self.config.word_bio_column)
        # check type of bio labels feature
        l = features[self.config.word_bio_column]
        if not (isinstance(l, Sequence) and isinstance(l.feature, ClassLabel)):
            raise TypeError("Expected bio labels to be a `Sequence` of `ClassLabels`, got %s." % l)

        # add feature
        return Features({
            self.config.output_column: Sequence(
                ClassLabel(names=l.feature.names), length=f.length
            )
        })

    def process(self, example:dict[str, Any]) -> dict[str, np.ndarray]:
        # get word ids from examples and compute special tokens mask
        word_ids = np.asarray(example[self.config.word_ids_column])
        special_tokens_mask = (word_ids < 0)

        # get token-level bio scheme and map it to word level
        bio = np.asarray(example[self.config.word_bio_column])
        bio = np.where(special_tokens_mask, self.config.ignore_label_index, bio[word_ids])
        # mask all tags that should be in-tags but are begin-tags
        in_mask = np.zeros_like(bio, dtype=bool)
        in_mask[1:] = (word_ids[:-1] == word_ids[1:])
        in_mask &= ~special_tokens_mask
        # convert all begin tags that should be in tags
        bio[in_mask] = self.begin2in[bio[in_mask]]

        # return token-level bio scheme
        return example | {self.config.output_column: bio}

class FromWordLevelSpans(DataProcessor):
    """Bio label processor backbone for generating token-level labels from word-level spans"""

    @property
    def entity_names(self) -> list[str]:
        # return entity names from input features
        return self.in_features[self.config.word_span_column]['type'].feature.names

    def bio_label2id(self, values:str|list[str]) -> int|list[int]:
        return self.out_features[self.config.output_column].feature.str2int(values)

    @property
    def out_tag_id(self) -> int:
        return self.bio_label2id(self.config.out_tag)

    def map_features(self, features:Features) -> Sequence:

        # make sure word ids are present in features
        if self.config.word_ids_column not in features:
            raise KeyError("`%s` not present in features!" % self.config.word_ids_column)
        # check type of word ids column
        f = features[self.config.word_ids_column]
        if not (isinstance(f, Sequence) and (f.feature == Value('int32'))):
            raise TypeError("Expected word ids to be a sequence of ints, got %s." % f)

        # check if word span column is present
        if self.config.word_span_column not in features:
            raise KeyError("`%s` not present in features!" % self.config.word_span_column)
        # check type of span column
        l = features[self.config.word_span_column]
        # TODO: check feature type (must containt begin, end, type of correct feature types)

        # build bio tags
        names = l['type'].feature.names
        bio_tags = [self.config.out_tag] + [
            "%s%s" % (prefix, name) for name in names for prefix in (
                self.config.begin_tag_prefix,
                self.config.in_tag_prefix
            )
        ]
        # add feature
        return Features({
            self.config.output_column: Sequence(
                ClassLabel(names=bio_tags), length=f.length
            )
        })

    def process(self, example:dict[str, Any]) -> dict[str, np.ndarray]:
        # get word ids from examples and compute special tokens mask
        word_ids = np.asarray(example[self.config.word_ids_column])
        special_tokens_mask = (word_ids < 0)

        # build initial empty bio labels and get spans
        bio = np.where(special_tokens_mask, self.config.ignore_label_index, self.out_tag_id)
        spans = example[self.config.word_span_column]

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

        # return token-level bio scheme
        return example | {self.config.output_column: bio}


class FromCharacterLevelSpans(DataProcessor):
    """Bio label processor backbone for generating token-level labels from character-level spans"""

    @property
    def entity_names(self) -> list[str]:
        # return entity names from input features
        return self.in_features[self.config.char_span_column]['type'].feature.names

    def bio_label2id(self, values:str|list[str]) -> int|list[int]:
        return self.out_features[self.config.output_column].feature.str2int(values)

    @property
    def out_tag_id(self) -> int:
        return self.bio_label2id(self.config.out_tag)

    def map_features(self, features:Features) -> Sequence:

        # make sure character offsets are present in features
        if self.config.char_offsets_column not in features:
            raise KeyError("`%s` not present in features!" % self.config.char_offsets_column)

        # check type of character offsets column
        f = features[self.config.char_offsets_column]
        # must be a sequence of (b, e)-tuples
        if not (
            isinstance(f, Sequence) and
            isinstance(f.feature, Sequence) and
            (f.feature.length == 2) and
            (f.feature.feature == Value('int32'))
        ):
            raise TypeError(
                "Expected character offsets to be a sequence of (begin, end)-tuples, got %s." % f
            )

        # check if character span column is present
        if self.config.char_span_column not in features:
            raise KeyError("`%s` not present in features!" % self.config.char_span_column)
        # check type of span column
        l = features[self.config.char_span_column]
        # TODO: check feature type (must containt begin, end, type of correct feature types)

        # build bio tags
        names = l['type'].feature.names
        bio_tags = [self.config.out_tag] + [
            "%s%s" % (prefix, name) for name in names for prefix in (
                self.config.begin_tag_prefix,
                self.config.in_tag_prefix
            )
        ]
        # add feature
        return Features({
            self.config.output_column: Sequence(
                ClassLabel(names=bio_tags), length=f.length
            )
        })

    def process(self, example:dict[str, Any]) -> dict[str, np.ndarray]:
        # get character offsets from example
        char_offsets = np.asarray(example[self.config.char_offsets_column])
        special_tokens_mask = (char_offsets == 0).all(axis=1)

        # build initial empty bio labels and get spans
        bio = np.where(special_tokens_mask, self.config.ignore_label_index, self.out_tag_id)
        spans = example[self.config.char_span_column]

        # process each span
        for entity_t, begin, end in zip(spans['type'], spans['begin'], spans['end']):
            # get entity mask
            mask = (begin <= char_offsets[:, 0]) & (char_offsets[:, 1] <= end)
            mask &= (~special_tokens_mask)
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

        # return token-level bio scheme
        return example | {self.config.output_column: bio}


class BioLabelProcessor(DataProcessor):
    """BIO Labels Processor"""

    def __init__(self, config:BioLabelProcessorConfig) -> None:
        super(BioLabelProcessor, self).__init__(config)
        # select the correct backbone given the config
        self.backbone = (
            FromWordLevelLabels(config) if config.word_bio_column is not None else
            FromWordLevelSpans(config) if config.word_span_column is not None else
            FromCharacterLevelSpans(config)
        )

    def map_features(self, features:Features) -> Features:
        return self.backbone.prepare(features)

    def process(self, example:dict[str, Any]) -> dict[str, np.ndarray]:
        return self.backbone.process(example)
