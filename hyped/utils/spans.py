import numpy as np
from typing import Sequence


def make_spans_exclusive(
    spans: Sequence[tuple[int]],
    is_inclusive: bool
) -> list[tuple[int]] | tuple[int]:
    """Convert arbitrary (inclusive or exclusive) spans
    to be exclusive.

    Arguments:
        spans (Sequence[tuple[int]]): sequence of spans to process
        is_inclusive (bool):
            bool indicating whether the given spans are inclusive or not

    Returns:
        exclusive_spans (list[tuple[int]] | tuple[int]):
            processed spans guaranteed to be exclusive
    """
    spans = np.asarray(list(spans)).reshape(-1, 2)
    spans[..., 1] += int(is_inclusive)
    return spans.tolist()


def compute_spans_overlap_matrix(
    source_spans: Sequence[tuple[int]],
    target_spans: Sequence[tuple[int]],
    is_source_inclusive: bool = False,
    is_target_inclusive: bool = False
) -> np.ndarray:
    """Compute the span overlap matrix

    The span overlap matrix `O` is a binary matrix of shape (n, m) where
    n is the number of source spans and m is the number of target
    spans. The boolean value `O_ij` indicates whether the i-th source
    span overlaps with the j-th target span.

    Arguments:
        source_spans (Sequence[tuple[int]]):
            either a sequence of source spans or a single source span
        target_spans (Sequence[tuple[int]]):
            either a sequence of target spans or a single target span
        is_source_inclusive (bool):
            bool indicating whether the source spans are inclusive or not
        is_target_inclusive (bool):
            bool indicating whether the target spans are inclusive or not

    Returns:
        O (np.ndarray): binary overlap matrix
    """
    # make all spans exclusive
    source_spans = make_spans_exclusive(source_spans, is_source_inclusive)
    target_spans = make_spans_exclusive(target_spans, is_target_inclusive)
    # convert spans to numpy arrays
    source_spans = np.asarray(source_spans).reshape(-1, 2)
    target_spans = np.asarray(target_spans).reshape(-1, 2)
    # compute overlap mask
    return (
        (source_spans[:, 0, None] <= target_spans[None, :, 0]) &
        (target_spans[None, :, 1] < source_spans[:, 1, None])
    )
