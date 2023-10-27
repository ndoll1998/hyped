from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    INDEX_TYPES,
    raise_feature_exists,
    raise_feature_is_sequence,
    raise_features_align,
    get_sequence_length,
)
from hyped.utils.spans import (
    make_spans_exclusive,
    resolve_overlaps,
    ResolveOverlapsStrategy,
)
from dataclasses import dataclass
from datasets import Features, Sequence, ClassLabel, Value
from typing import Any, Literal


@dataclass
class BioTaggerConfig(BaseDataProcessorConfig):
    """Begin-In-Out (BIO) Tagger Config

    Convert Entity span annotations to per-token labels
    using the BIO-tagging scheme.

    Attributes:
        begin_tag_prefix (str):
            tag prefix used to mark the beginning of a new entity
            of a specific class
        in_tag_prefix (str):
            tag prefix used to mark the interior of an entity
            of a specific class
        out_tag (str):
            out tag used to mark tokens that are not part of any entity
        input_sequence (str): column containing the input sequence
        entity_spans_begin (str):
            column containing begins of the entity span annotations
        entity_spans_end (str):
            column containing ends of the entity span annotations
        entity_spans_label (str):
            column containing the entity class label to each entity
            span
        entity_spans_inclusive (bool):
            whether the end coordinate of the entity spans are
            inclusive or exclusive. Defaults to false.
        resolve_overlaps (ResolveOverlapsStrategy):
            the filter strategy to apply on overlapping entities. By default
            set to `ResolveOverlapsStrategy.RAISE` which will raise an
            Exception when an overlap is detected.
            See `hyped.utils.spans.ResolveOverlapsStrategy` for other options.
    """

    t: Literal[
        "hyped.data.processors.taggers.bio"
    ] = "hyped.data.processors.taggers.bio"

    begin_tag_prefix: str = "B-"
    in_tag_prefix: str = "I-"
    out_tag: str = "O"

    input_sequence: str = None
    entity_spans_begin: str = None
    entity_spans_end: str = None
    entity_spans_label: str = None

    entity_spans_inclusive: bool = False
    # handle overlaps between spans
    resolve_overlaps: ResolveOverlapsStrategy = ResolveOverlapsStrategy.RAISE


class BioTagger(BaseDataProcessor[BioTaggerConfig]):
    """Begin-In-Out (BIO) Tagger Config

    Convert Entity span annotations to per-token labels
    using the BIO-tagging scheme.
    """

    def _tag_sequence_feature(self, features: Features) -> Sequence:
        # the entity class label feature must be a sequence of
        # string values or class labels
        raise_feature_is_sequence(
            self.config.entity_spans_label,
            features[self.config.entity_spans_label],
            [Value("string"), ClassLabel],
        )

        # get the item feature type of the sequence
        feature = features[self.config.entity_spans_label].feature

        # get the length of the input sequence
        length = get_sequence_length(features[self.config.input_sequence])

        # build output feature type
        if isinstance(feature, ClassLabel):
            bio_feature_type = ClassLabel(
                names=[self.config.out_tag]
                + [
                    "%s%s" % (prefix, label)
                    for label in feature.names
                    for prefix in [
                        self.config.begin_tag_prefix,
                        self.config.in_tag_prefix,
                    ]
                ]
            )

            return Sequence(bio_feature_type, length=length)

        elif feature == Value("string"):
            return Sequence(Value("string"), length=length)

    @property
    def entity_label_space(self) -> None | ClassLabel:
        feature = self.in_features[self.config.entity_spans_label].feature
        return feature if isinstance(feature, ClassLabel) else None

    @property
    def bio_label_space(self) -> None | ClassLabel:
        feature = self.new_features["bio_tags"].feature
        return feature if isinstance(feature, ClassLabel) else None

    def map_features(self, features: Features) -> Features:
        # make sure the input sequence exists and is a sequence
        raise_feature_exists(self.config.input_sequence, features)
        raise_feature_is_sequence(
            self.config.input_sequence, features[self.config.input_sequence]
        )

        # make sure entity spans exist and are of correct type
        raise_feature_exists(self.config.entity_spans_begin, features)
        raise_feature_exists(self.config.entity_spans_end, features)
        # begin and end sequences should contain indices
        raise_feature_is_sequence(
            self.config.entity_spans_begin,
            features[self.config.entity_spans_begin],
            INDEX_TYPES,
        )
        raise_feature_is_sequence(
            self.config.entity_spans_end,
            features[self.config.entity_spans_end],
            INDEX_TYPES,
        )
        raise_features_align(
            self.config.entity_spans_begin,
            self.config.entity_spans_end,
            features[self.config.entity_spans_begin],
            features[self.config.entity_spans_end],
        )

        return {"bio_tags": self._tag_sequence_feature(features)}

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        # get length of input sequence
        length = len(example[self.config.input_sequence])

        # get entity spans
        spans = zip(
            example[self.config.entity_spans_begin],
            example[self.config.entity_spans_end],
        )
        # make entity spans exclusive and filter overlapping spans
        init_spans = make_spans_exclusive(
            spans, self.config.entity_spans_inclusive
        )
        spans = resolve_overlaps(init_spans, self.config.resolve_overlaps)

        # get the entity labels
        labels = example[self.config.entity_spans_label]
        # convert label ids to label strings
        if self.entity_label_space is not None:
            labels = self.entity_label_space.int2str(labels)

        # if any spans have been filtered then
        # the labels need to be filtered to
        if len(spans) < len(init_spans):
            labels = map(labels.__getitem__, map(init_spans.index, spans))

        # build initial tag sequence of all out tags
        tags = [self.config.out_tag] * length

        # insert all entity spans
        for label, (b, e) in zip(labels, spans):
            # all overlaps should be filtered at this point
            assert all(t == self.config.out_tag for t in tags[b:e])
            # add entity to tag sequence
            tags[b:e] = ["%s%s" % (self.config.in_tag_prefix, label)] * (e - b)
            tags[b] = "%s%s" % (self.config.begin_tag_prefix, label)

        # convert label strings to label ids
        if self.bio_label_space is not None:
            tags = self.bio_label_space.str2int(tags)

        # return bio tags
        return {"bio_tags": tags}
