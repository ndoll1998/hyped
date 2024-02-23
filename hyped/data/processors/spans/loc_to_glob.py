from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from datasets import Features

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
    raise_feature_is_sequence,
    raise_features_align,
)
from hyped.utils.spans import make_spans_exclusive

from .outputs import SpansOutputs


@dataclass
class LocalToGlobalOffsetsConfig(BaseDataProcessorConfig):
    """Offset Conversion Data Processor

    Convert local offsets to global offsets. Useful when
    tokenizing pre-tokenized words and needing the global
    token offsets.

    Specifically the i-th token span is computed as follows:

        span_begin[i] = (
            local_offsets_begin[i] + global_offsets_begin[
                local_to_global_mapping[i]
            ]
        )

        span_end[i] = (
            local_offsets_end[i] + global_offsets_begin[
                local_to_global_mapping[i]
            ]
        )

    Type Identifier: `hyped.data.processors.spans.local_to_global_offsets`

    Attributes:
        local_offsets_begin (FeatureKey):
            feature containing begins of local offsets
        local_offsets_end (FeatureKey):
            feature containing ends of local offsets
        global_offsets_begin (None | FeatureKey):
            feature containing the global character-level begin
            offsets of each local instance (i.e. word). When set
            to None, each local instance will be offset by the
            current accumulated text length + 1.
        local_to_global_mapping (None | FeatureKey):
            feature containing a sequence of integers where the
            i-th position is the index of the global offset element
            by which to offset the i-th local offset. Required when
            `global_offsets_begin` is specified.
        local_offsets_inclusive (bool):
            bool indicating whether the local offset spans are inclusive
            or not. Defaults to False.
    """

    t: Literal[
        "hyped.data.processors.spans.local_to_global_offsets"
    ] = "hyped.data.processors.spans.local_to_global_offsets"

    # local offsets
    local_offsets_begin: FeatureKey = None
    local_offsets_end: FeatureKey = None
    # map to global offset
    global_offsets_begin: None | FeatureKey = None
    local_to_global_mapping: None | FeatureKey = None
    # offsets inclusive or not
    local_offsets_inclusive: bool = False


class LocalToGlobalOffsets(BaseDataProcessor[LocalToGlobalOffsetsConfig]):
    """Offset Conversion Data Processor

    Convert local offsets to global offsets. Useful when
    tokenizing pre-tokenized words and needing the global
    token offsets.

    Specifically the i-th token span is computed as follows:

        span_begin[i] = (
            local_offsets_begin[i] + global_offsets_begin[
                local_to_global_mapping[i]
            ]
        )

        span_end[i] = (
            local_offsets_end[i] + global_offsets_begin[
                local_to_global_mapping[i]
            ]
        )
    """

    def map_features(self, features: Features) -> Features:
        """Check input features and return feature mapping
        for global offsets.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): global offset features
        """
        # make sure local offsets features exist
        local_offsets_begin = get_feature_at_key(
            features, self.config.local_offsets_begin
        )
        local_offsets_end = get_feature_at_key(
            features, self.config.local_offsets_end
        )
        # and are of the correct type
        raise_feature_is_sequence(
            self.config.local_offsets_begin,
            local_offsets_begin,
            INDEX_TYPES,
        )
        raise_feature_is_sequence(
            self.config.local_offsets_end,
            local_offsets_end,
            INDEX_TYPES,
        )
        # make sure begin and end features match exactly
        raise_features_align(
            self.config.local_offsets_begin,
            self.config.local_offsets_end,
            local_offsets_begin,
            local_offsets_end,
        )

        if self.config.global_offsets_begin is not None:
            if self.config.local_to_global_mapping is None:
                raise ValueError(
                    "`local_to_global_mapping` argument not specified"
                )

            global_offsets_begin = get_feature_at_key(
                features, self.config.global_offsets_begin
            )
            local_to_global_mapping = get_feature_at_key(
                features, self.config.local_to_global_mapping
            )

            raise_feature_is_sequence(
                self.config.global_offsets_begin,
                global_offsets_begin,
                INDEX_TYPES,
            )
            raise_feature_is_sequence(
                self.config.local_to_global_mapping,
                local_to_global_mapping,
                INDEX_TYPES,
            )

            raise_features_align(
                self.config.local_to_global_mapping,
                self.config.local_offsets_begin,
                local_to_global_mapping,
                local_offsets_begin,
            )

        return Features(
            {
                SpansOutputs.BEGINS.value: local_offsets_begin,
                SpansOutputs.ENDS.value: local_offsets_end,
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
            out (dict[str, Any]): global offsets
        """

        # get local offset spans
        local_offsets = zip(
            get_value_at_key(example, self.config.local_offsets_begin),
            get_value_at_key(example, self.config.local_offsets_end),
        )
        # make offsets exclusive and convert to numpy array
        local_offsets = make_spans_exclusive(
            local_offsets, self.config.local_offsets_inclusive
        )
        local_offsets = np.asarray(local_offsets).reshape(-1, 2)

        if self.config.global_offsets_begin is None:
            mask = local_offsets[:, 0] == 0
            # increase the global index by one whenever the local
            # offsets begin jumps back to zero
            # -1 to account for the initial zero
            local_to_global_mapping = (mask).cumsum() - 1

            # global offsets are cumulative sums ends of all
            # local instances + 1 to add a space in between
            mask = np.roll(mask, shift=-1)
            mask[-1] = False
            global_offsets_begin = (local_offsets[mask, 1] + 1).cumsum()
            # insert initial zero
            global_offsets_begin = np.insert(global_offsets_begin, 0, 0)

        else:
            # get global information
            global_offsets_begin = np.asarray(
                get_value_at_key(example, self.config.global_offsets_begin)
            )
            local_to_global_mapping = np.asarray(
                get_value_at_key(example, self.config.local_to_global_mapping)
            )

        # compute offsets on global scale
        offsets_begin = (
            local_offsets[:, 0] + global_offsets_begin[local_to_global_mapping]
        )
        offsets_end = (
            local_offsets[:, 1] + global_offsets_begin[local_to_global_mapping]
        )
        # return offsets
        return {
            SpansOutputs.BEGINS.value: offsets_begin.tolist(),
            SpansOutputs.ENDS.value: offsets_end.tolist(),
        }
