from .outputs import SpansOutputs
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    INDEX_TYPES,
    raise_features_align,
    raise_feature_is_sequence,
    check_feature_equals,
    check_feature_is_sequence,
)
from hyped.utils.feature_access import (
    FeatureKey,
    get_feature_at_key,
    get_value_at_key,
)
from hyped.utils.spans import compute_spans_overlap_matrix
from datasets import Features
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class CoveredIndexSpansConfig(BaseDataProcessorConfig):
    """Covered Index Span Data Processor Config

    Let q = (b, e) be a query span and A = [(b_k, e_k)]_k be an
    ordered span sequence in the same domain. Then the processor
    finds the index span (i, j) of the query span in A. That is
    the index span fulfills the following equation:

        q = ((A_i)_0, (A_(j-1))_1)

    Note that the output spans are exclusive, i.e. the index of
    the last member to the span is j-1.

    A common usecase of this operation is the conversion of
    character-level span annotations to token-level spans, as
    typically required in data processing for squad-style
    Question Answering (QA) or Named-Entity-Recognition (NER).

    Type Identifier: `hyped.data.processors.spans.covered_idx_spans`

    Attributes:
        queries_begin (FeatureKey):
            input feature containing the begin value(s) of the query
            span. Can be either a single value or a sequence of values.
        queries_end (FeatureKey):
            input feature containing the end value(s) of the query
            span. Can be either a single value or a sequence of values.
        spans_begin (FeatureKey):
            input feature containing the begin values of the span sequence A.
        spans_end (FeatureKey):
            input feature containing the end values of the span sequence A.
        is_queries_inclusive (bool):
            whether the end coordinates of the query span(s) are
            inclusive or exclusive. Defaults to false.
        is_spans_inclusive (bool):
            whether the end coordinates of the spans in the sequence A are
            inclusive or exclusive. Defaults to false.
    """

    t: Literal[
        "hyped.data.processors.spans.covered_idx_spans"
    ] = "hyped.data.processors.spans.covered_idx_spans"

    # query spans
    queries_begin: FeatureKey = None
    queries_end: FeatureKey = None
    # span sequence
    spans_begin: FeatureKey = None
    spans_end: FeatureKey = None
    # whether the end coordinates are inclusive of exclusive
    is_queries_inclusive: bool = False
    is_spans_inclusive: bool = False


class CoveredIndexSpans(BaseDataProcessor[CoveredIndexSpansConfig]):
    """Covered Index Span Data Processor Config

    Let q = (b, e) be a query span and A = [(b_k, e_k)]_k be an
    ordered span sequence in the same domain. Then the processor
    finds the index span (i, j) of the query span in A. That is
    the index span fulfills the following equation:

        q = ((A_i)_0, (A_(j-1))_1)

    Note that the output spans are exclusive, i.e. the index of
    the last member to the span is j-1.

    A common usecase of this operation is the conversion of
    character-level span annotations to token-level spans, as
    typically required in data processing for squad-style
    Question Answering (QA) or Named-Entity-Recognition (NER).
    """

    def map_features(self, features: Features) -> Features:
        """Check input features and return feature mapping
        for token-level span annotations.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): token-level span annotation features
        """
        # get all features
        queries_begin = get_feature_at_key(features, self.config.queries_begin)
        queries_end = get_feature_at_key(features, self.config.queries_end)
        spans_begin = get_feature_at_key(features, self.config.spans_begin)
        spans_end = get_feature_at_key(features, self.config.spans_end)

        # character spans must either be a sequence of
        # integers or an integer value
        for key, feature in [
            (self.config.queries_begin, queries_begin),
            (self.config.queries_end, queries_end),
        ]:
            if not (
                check_feature_is_sequence(feature, INDEX_TYPES)
                or check_feature_equals(feature, INDEX_TYPES)
            ):
                raise TypeError(
                    "Expected `%s` to be an integer value or a sequence "
                    "of integers, got %s" % (key, feature)
                )

        # character spans begin and end features must align
        raise_features_align(
            self.config.queries_begin,
            self.config.queries_end,
            queries_begin,
            queries_end,
        )

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

        return Features(
            {
                SpansOutputs.BEGINS.value: queries_begin,
                SpansOutputs.ENDS.value: queries_end,
            }
        )

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Apply processor to an example

        Arguments:
            example (dict[str, Any]): example to process
            index (int): dataset index of the example
            rank (int): execution process rank

        Returns:
            out (dict[str, Any]): token-level span annotations
        """
        # get query spans
        queries_begin = get_value_at_key(example, self.config.queries_begin)
        queries_end = get_value_at_key(example, self.config.queries_end)
        # check if it is a single value
        is_value = isinstance(queries_begin, int)
        # pack value into list
        if is_value:
            queries_begin = [queries_begin]
            queries_end = [queries_end]

        # build spans
        queries = zip(queries_begin, queries_end)
        spans = zip(
            get_value_at_key(example, self.config.spans_begin),
            get_value_at_key(example, self.config.spans_end),
        )

        # for each query span find the  spans that it overlaps with
        mask = compute_spans_overlap_matrix(
            queries,
            spans,
            self.config.is_queries_inclusive,
            self.config.is_spans_inclusive,
        )

        # get begins and ends from mask
        idx_spans_begin = mask.argmax(axis=1)
        idx_spans_end = idx_spans_begin + mask.sum(axis=1)
        # convert to list
        idx_spans_begin = idx_spans_begin.tolist()
        idx_spans_end = idx_spans_end.tolist()

        # return value when input was also a single value
        if is_value:
            idx_spans_begin = idx_spans_begin[0]
            idx_spans_end = idx_spans_end[0]

        # return index spans
        return {
            SpansOutputs.BEGINS.value: idx_spans_begin,
            SpansOutputs.ENDS.value: idx_spans_end,
        }
