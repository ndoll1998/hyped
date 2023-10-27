from tests.data.processors.base import BaseTestDataProcessor
from hyped.data.processors.taggers.bio import (
    BioTagger,
    BioTaggerConfig,
)
from hyped.utils.spans import ResolveOverlapsStrategy
from datasets import Features, Sequence, Value, ClassLabel
from itertools import permutations
import pytest


class TestBioTagger(BaseTestDataProcessor):
    @pytest.fixture(params=[False, True])
    def is_span_inclusive(self, request):
        return request.param

    @pytest.fixture(params=[-1, 8, 16, 32])
    def length(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [Value("string"), Value("string")],
            [
                ClassLabel(names=list("XY")),
                ClassLabel(names=["O", "B-X", "I-X", "B-Y", "I-Y"]),
            ],
        ]
    )
    def labels_and_tags_feature(self, request):
        return request.param

    @pytest.fixture
    def labels_feature(self, labels_and_tags_feature):
        return labels_and_tags_feature[0]

    @pytest.fixture
    def tags_feature(self, labels_and_tags_feature):
        return labels_and_tags_feature[1]

    @pytest.fixture
    def in_features(self, length, labels_feature):
        return Features(
            {
                "input_sequence": Sequence(Value("int32"), length=length),
                "entity_spans_begin": Sequence(Value("int32")),
                "entity_spans_end": Sequence(Value("int32")),
                "entity_spans_label": Sequence(labels_feature),
            }
        )

    @pytest.fixture(params=permutations(range(4)))
    def batch(self, request, length, labels_feature, is_span_inclusive):
        # set length if it is undefined
        length = 20 if length == -1 else length

        # define sample entities
        x, y = "X", "Y"
        entity_spans_begin = [3, 14, 24, 28]
        entity_spans_end = [8, 19, 25, 31]
        entity_spans_label = [x, y, y, x]
        # permute entities
        perm = request.param
        entity_spans_begin = [entity_spans_begin[i] for i in perm]
        entity_spans_end = [entity_spans_end[i] for i in perm]
        entity_spans_label = [entity_spans_label[i] for i in perm]
        # remove out of bounds entities
        mask = [i < length for i in entity_spans_begin]
        entity_spans_begin = [i for i, v in zip(entity_spans_begin, mask) if v]
        entity_spans_end = [i for i, v in zip(entity_spans_end, mask) if v]
        entity_spans_label = [l for l, v in zip(entity_spans_label, mask) if v]
        # truncate entities at sequence length
        entity_spans_end = [min(i, length) for i in entity_spans_end]
        # apply inclusive or not
        entity_spans_end = [
            i - int(is_span_inclusive) for i in entity_spans_end
        ]

        # convert to label indices
        if isinstance(labels_feature, ClassLabel):
            entity_spans_label = labels_feature.str2int(entity_spans_label)

        # return batch
        return {
            "input_sequence": [list(range(length))],
            "entity_spans_begin": [entity_spans_begin],
            "entity_spans_end": [entity_spans_end],
            "entity_spans_label": [entity_spans_label],
        }

    @pytest.fixture
    def processor(self, is_span_inclusive):
        return BioTagger(
            BioTaggerConfig(
                input_sequence="input_sequence",
                entity_spans_begin="entity_spans_begin",
                entity_spans_end="entity_spans_end",
                entity_spans_label="entity_spans_label",
                entity_spans_inclusive=is_span_inclusive,
            )
        )

    @pytest.fixture
    def expected_out_features(self, length, tags_feature):
        return Features({"bio_tags": Sequence(tags_feature, length=length)})

    @pytest.fixture
    def expected_out_batch(self, length, tags_feature):
        # initial bio tags are all out tags
        tags = ["O"] * 32
        # X entities
        tags[3] = "B-X"
        tags[4:8] = ["I-X"] * 4
        tags[28] = "B-X"
        tags[29:31] = ["I-X"] * 2
        # Y entites
        tags[14] = "B-Y"
        tags[15:19] = ["I-Y"] * 4
        tags[24] = "B-Y"

        # cut tags at relevant length
        tags = tags[: (20 if length == -1 else length)]
        # convert to label indices
        if isinstance(tags_feature, ClassLabel):
            tags = tags_feature.str2int(tags)

        # return bio tags
        return {"bio_tags": [tags]}


class TestBioTaggerWithOverlaps(TestBioTagger):
    @pytest.fixture(
        params=[
            ResolveOverlapsStrategy.RAISE,
            ResolveOverlapsStrategy.KEEP_SMALLEST,
            ResolveOverlapsStrategy.KEEP_LARGEST,
        ]
    )
    def strategy(self, request):
        return request.param

    @pytest.fixture
    def processor(self, is_span_inclusive, strategy):
        return BioTagger(
            BioTaggerConfig(
                input_sequence="input_sequence",
                entity_spans_begin="entity_spans_begin",
                entity_spans_end="entity_spans_end",
                entity_spans_label="entity_spans_label",
                entity_spans_inclusive=is_span_inclusive,
                resolve_overlaps=strategy,
            )
        )

    @pytest.fixture
    def batch(self, length, labels_feature, is_span_inclusive):
        # set length if it is undefined
        length = 20 if length == -1 else length

        # define sample entities
        x, y = "X", "Y"
        entity_spans_begin = [3, 14, 24, 28, 1]
        entity_spans_end = [8, 19, 25, 31, 15]
        entity_spans_label = [x, y, y, x, x]
        # remove out of bounds entities
        mask = [i < length for i in entity_spans_begin]
        entity_spans_begin = [i for i, v in zip(entity_spans_begin, mask) if v]
        entity_spans_end = [i for i, v in zip(entity_spans_end, mask) if v]
        entity_spans_label = [l for l, v in zip(entity_spans_label, mask) if v]
        # truncate entities at sequence length
        entity_spans_end = [min(i, length) for i in entity_spans_end]
        # apply inclusive or not
        entity_spans_end = [
            i - int(is_span_inclusive) for i in entity_spans_end
        ]

        # convert to label indices
        if isinstance(labels_feature, ClassLabel):
            entity_spans_label = labels_feature.str2int(entity_spans_label)

        # return batch
        return {
            "input_sequence": [list(range(length))],
            "entity_spans_begin": [entity_spans_begin],
            "entity_spans_end": [entity_spans_end],
            "entity_spans_label": [entity_spans_label],
        }

    @pytest.fixture
    def expected_err_on_process(self, strategy):
        if strategy == ResolveOverlapsStrategy.RAISE:
            return ValueError

    @pytest.fixture
    def expected_out_batch(self, strategy, length, tags_feature):
        if strategy == ResolveOverlapsStrategy.RAISE:
            # no output expected in this case
            return None

        # initial bio tags are all out tags
        tags = ["O"] * 32

        if strategy == ResolveOverlapsStrategy.KEEP_SMALLEST:
            # X entities
            tags[3] = "B-X"
            tags[4:8] = ["I-X"] * 4
            # Y entites
            tags[14] = "B-Y"
            tags[15:19] = ["I-Y"] * 4

        if strategy == ResolveOverlapsStrategy.KEEP_LARGEST:
            # overlapping X entity
            tags[1] = "B-X"
            tags[2:15] = ["I-X"] * 13

        # others
        tags[24] = "B-Y"
        tags[28] = "B-X"
        tags[29:31] = ["I-X"] * 2

        # cut tags at relevant length
        tags = tags[: (20 if length == -1 else length)]

        # convert to label indices
        if isinstance(tags_feature, ClassLabel):
            tags = tags_feature.str2int(tags)

        # return bio tags
        return {"bio_tags": [tags]}
