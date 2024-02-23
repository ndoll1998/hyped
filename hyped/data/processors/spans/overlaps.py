from dataclasses import dataclass
from itertools import compress
from typing import Any, Literal

from datasets import Features, Sequence, Value

from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_access import (
    FeatureKey,
    get_feature_at_key,
    get_value_at_key,
)
from hyped.utils.feature_checks import (
    INDEX_TYPES,
    get_sequence_feature,
    get_sequence_length,
    raise_feature_is_sequence,
    raise_features_align,
)
from hyped.utils.spans import (
    ResolveOverlapsStrategy,
    make_spans_exclusive,
    resolve_overlaps,
)


@dataclass
class ResolveSpanOverlapsConfig(BaseDataProcessorConfig):
    """Resolve Span Overlaps Data Processor Config

    Resolve overlaps between spans of a span sequence.

    Type Identifier: `hyped.data.processors.spans.covered_idx_spans`

    Attributes:
        spans_begin (FeatureKey):
            input feature containing the begin values of the span sequence A.
        spans_end (FeatureKey):
            input feature containing the end values of the span sequence A.
        is_spans_inclusive (bool):
            whether the end coordinates of the spans in the sequence A are
            inclusive or exclusive. Defaults to false.
        strategy (ResolveOverlapsStrategy):
            the strategy to apply when resolving the overlaps. Defaults to
            `ResolveOverlapsStrategy.APPROX` which aims to minimize the
            number of spans to remove. For other options please refer to
            `hyped.utils.spans.ResolveOverlapsStrategy`.
    """

    t: Literal[
        "hyped.data.processors.spans.resolve_overlaps"
    ] = "hyped.data.processors.spans.resolve_overlaps"

    # span sequence
    spans_begin: FeatureKey = None
    spans_end: FeatureKey = None
    is_spans_inclusive: bool = False
    # strategy to apply
    strategy: ResolveOverlapsStrategy = ResolveOverlapsStrategy.APPROX


class ResolveSpanOverlaps(BaseDataProcessor[ResolveSpanOverlapsConfig]):
    """Resolve Span Overlaps Data Processor

    Resolve overlaps between spans of a span sequence.
    """

    def map_features(self, features: Features) -> Features:
        """Check input features and overwrite the given
        span sequence. Also returns a mask over the initial span
        sequence indicating which spans of the sequence where kept.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): output feature mapping
        """
        # make sure all features exist
        spans_begin = get_feature_at_key(features, self.config.spans_begin)
        spans_end = get_feature_at_key(features, self.config.spans_end)
        # spans must be sequence of integers
        raise_feature_is_sequence(
            self.config.spans_begin,
            spans_begin,
            INDEX_TYPES,
        )
        raise_feature_is_sequence(
            self.config.spans_begin,
            spans_end,
            INDEX_TYPES,
        )
        # and they must align excatly
        raise_features_align(
            self.config.spans_begin,
            self.config.spans_end,
            spans_begin,
            spans_end,
        )
        # get item feature and length from span sequence feature
        feature = get_sequence_feature(spans_begin)
        length = get_sequence_length(spans_begin)
        # returns a mask over the span sequence and overwrite
        # the span sequence
        return {
            "resolve_overlaps_mask": Sequence(Value("bool"), length=length),
            self.config.spans_begin: Sequence(feature),
            self.config.spans_end: Sequence(feature),
        }

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Apply processor to an example

        Arguments:
            example (dict[str, Any]): example to process
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]): spans without overlaps
        """

        spans = list(
            zip(
                get_value_at_key(example, self.config.spans_begin),
                get_value_at_key(example, self.config.spans_end),
            )
        )

        if len(spans) == 0:
            # handle edgecase no spans
            return {
                "resolve_overlaps_mask": [],
                self.config.spans_begin: [],
                self.config.spans_end: [],
            }

        # make spans exclusive and resolve overlaps
        excl_spans = make_spans_exclusive(
            spans, self.config.is_spans_inclusive
        )
        mask = resolve_overlaps(excl_spans, strategy=self.config.strategy)
        # apply mask to spans and return features
        spans = compress(spans, mask)
        spans_begin, spans_end = zip(*spans)

        return {
            "resolve_overlaps_mask": mask,
            self.config.spans_begin: list(spans_begin),
            self.config.spans_end: list(spans_end),
        }
