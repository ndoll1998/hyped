import numpy as np
from enum import StrEnum
from typing import Sequence


class ResolveOverlapsStrategy(StrEnum):
    """Enum over strategies to apply when resolving overlapping strategies
    Used as an argument to `hyped.utils.resolve_overlaps`.
    """

    APPROX = "approx"
    """Approximate the largest non-overlapping subset
    of spans"""

    RAISE = "raise"
    """Raise a ValueError when an overlap is detected"""

    KEEP_FIRST = "keep_first"
    """When two or more spans overlap, keep the first
    one in the sequence of spans"""

    KEEP_LAST = "keep_last"
    """When two or more spans overlap, keep the last
    one in the sequence of spans"""

    KEEP_LARGEST = "keep_largest"
    """When two or more spans overlap, keep the largest
    of the overlapping spans"""

    KEEP_SMALLEST = "keep_smallest"
    """When two or more spans overlap, keep the smallest
    of the overlapping spans"""


def make_spans_exclusive(
    spans: Sequence[tuple[int]], is_inclusive: bool
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
    return list(map(tuple, spans.tolist()))


def compute_spans_overlap_matrix(
    source_spans: Sequence[tuple[int]],
    target_spans: Sequence[tuple[int]],
    is_source_inclusive: bool = False,
    is_target_inclusive: bool = False,
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
        (
            # source overlaps with target begin
            (source_spans[:, 0, None] <= target_spans[None, :, 0])
            & (target_spans[None, :, 0] < source_spans[:, 1, None])
        )
        | (
            # source overlaps with target end
            (source_spans[:, 0, None] < target_spans[None, :, 1])
            & (target_spans[None, :, 1] <= source_spans[:, 1, None])
        )
        | (
            # target is contained in source
            (source_spans[:, 0, None] <= target_spans[None, :, 0])
            & (target_spans[None, :, 1] <= source_spans[:, 1, None])
        )
        | (
            # source is contained in target
            (target_spans[None, :, 0] <= source_spans[:, 0, None])
            & (source_spans[:, 1, None] <= target_spans[None, :, 1])
        )
    )


def resolve_overlaps(
    spans: Sequence[tuple[int]],
    strategy: ResolveOverlapsStrategy = ResolveOverlapsStrategy.APPROX,
) -> list[tuple[int]]:
    """Resolve span overlaps

    Iteratively removes the span which overlaps with most other
    spans in the given sequence, while satisfying the resolve
    strategy. See `ResolveOverlapsStrategy` for more information.

    Arguments:
        spans (Sequence[tuple[int]]):
            sequence of potentially overlapping spans
        strategy (ResloveOverlapsStrategy):
            strategy to apply for resolving overlaps between spans
    """

    spans = np.asarray(list(spans)).reshape(-1, 2)
    # for each span find the spans it overlaps with
    overlap = compute_spans_overlap_matrix(spans, spans)
    counts = overlap.sum(axis=1)

    while (counts > 1).any():
        # every span in the overlap group is a potential candidate
        # there have to be at least two spans that overlap eachother
        cand_mask = overlap[counts.argmax(), :].copy()
        assert cand_mask.sum() >= 2

        if strategy == ResolveOverlapsStrategy.RAISE:
            raise ValueError("Detected Overlaps between spans")

        elif strategy == ResolveOverlapsStrategy.APPROX:
            pass

        elif strategy == ResolveOverlapsStrategy.KEEP_FIRST:
            # get the first index in the group and remove it
            # from the candidates
            first_idx = cand_mask.argmax()
            cand_mask[first_idx] = False

        elif strategy == ResolveOverlapsStrategy.KEEP_LAST:
            # get the last index in the group and remove it
            # from the candidate list
            last_idx = cand_mask.nonzero()[0][-1]
            cand_mask[last_idx] = False

        elif strategy == ResolveOverlapsStrategy.KEEP_LARGEST:
            # compute span sizes for all candidates
            cand_spans = spans[cand_mask]
            cand_sizes = cand_spans[:, 1] - cand_spans[:, 0]
            # make sure there remains at least one candidate
            if (cand_sizes < cand_sizes.max()).any():
                # keep only the smaller candidates
                cand_mask[cand_mask] &= cand_sizes < cand_sizes.max()

        elif strategy == ResolveOverlapsStrategy.KEEP_SMALLEST:
            # compute span sizes for all candidates
            cand_spans = spans[cand_mask]
            cand_sizes = cand_spans[:, 1] - cand_spans[:, 0]
            # make sure there remains at least one candidate
            if (cand_sizes > cand_sizes.min()).any():
                # keep only the smaller candidates
                cand_mask[cand_mask] &= cand_sizes > cand_sizes.min()

        # of all candidates select the one that overlaps
        # with the most entities
        cand_mask[cand_mask] = counts[cand_mask] == counts[cand_mask].max()
        assert cand_mask.any()
        # remove the first candidate
        idx_to_remove = cand_mask.argmax()

        # remove the span at the selected index
        spans = np.delete(spans, idx_to_remove, axis=0)
        # update counts
        counts = counts - overlap[idx_to_remove, :].astype(int)
        counts = np.delete(counts, idx_to_remove)
        # update the overlap matrix
        overlap = np.delete(overlap, idx_to_remove, axis=0)
        overlap = np.delete(overlap, idx_to_remove, axis=1)

    return list(map(tuple, spans.tolist()))
