from hyped.utils.spans import (
    make_spans_exclusive,
    compute_spans_overlap_matrix
)
import numpy as np
import pytest


class TestMakeSpansExclusive:

    @pytest.mark.parametrize(
        "spans, is_inclusive, expected_out", [
            [[], False, []],
            [[(0, 4)], False, [(0, 4)]],
            [[(0, 4)], True,  [(0, 5)]],
            [[(0, 4), (8, 12)], False, [(0, 4), (8, 12)]],
            [[(0, 4), (8, 12)], True,  [(0, 5), (8, 13)]],
        ]
    )
    def test(self, spans, is_inclusive, expected_out):
        assert make_spans_exclusive(spans, is_inclusive) == expected_out


class TestComputeSpansOverlapMatrix:

    def test(self):
        # test spans
        src_spans = [(0, 4), (5, 8), (15, 21)]
        tgt_spans = [(0, 4), (3, 7)]
        # expected overlap mask
        expected_mask = np.asarray([
            [True, True],
            [False, True],
            [False, False]
        ])
        # test
        assert (
            compute_spans_overlap_matrix(src_spans, tgt_spans) == expected_mask
        ).all()

