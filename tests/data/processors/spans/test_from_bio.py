from tests.data.processors.base import BaseTestDataProcessor
from hyped.data.processors.spans.outputs import LabelledSpansOutputs
from hyped.data.processors.spans.from_bio import (
    TokenSpansFromBioTags,
    TokenSpansFromBioTagsConfig,
)
from datasets import Features, Sequence, Value, ClassLabel
import pytest


class TestTokenSpansFromBioTags(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            # basics
            [(3, 4, "X")],
            [(3, 7, "X")],
            [(3, 7, "X"), (14, 15, "Y")],
            [(3, 7, "X"), (14, 15, "Y"), (16, 19, "Y"), (23, 29, "X")],
            # entities follow each other immediately
            [(3, 7, "X"), (7, 15, "Y")],
            [(2, 3, "Y"), (3, 7, "X"), (7, 15, "Y"), (15, 19, "X")],
            # entities sequence at boundaties, note length=32
            [(0, 7, "X"), (23, 29, "Y")],
            [(3, 7, "X"), (23, 32, "Y")],
            [(0, 7, "X"), (23, 32, "Y")],
            [(0, 7, "X"), (13, 19, "X"), (23, 32, "Y")],
        ]
    )
    def spans(self, request):
        return request.param

    @pytest.fixture(
        params=[
            [Value("string"), Value("string")],
            [
                ClassLabel(names=["X", "Y"]),
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
    def in_features(self, tags_feature):
        return Features({"bio_tags": Sequence(tags_feature)})

    @pytest.fixture
    def in_batch(self, spans, tags_feature):
        tags = ["O"] * 32
        # add entity spans
        for b, e, label in spans:
            tags[b + 1 : e] = ["I-%s" % label] * (e - b - 1)  # noqa: E203
            tags[b] = "B-%s" % label
        # convert to ids if needed
        if isinstance(tags_feature, ClassLabel):
            tags = tags_feature.str2int(tags)

        return {"bio_tags": [tags]}

    @pytest.fixture
    def processor(self):
        return TokenSpansFromBioTags(
            TokenSpansFromBioTagsConfig(bio_tags="bio_tags")
        )

    @pytest.fixture
    def expected_out_feature(self, labels_feature):
        return Features(
            {
                LabelledSpansOutputs.BEGINS: Sequence(Value("int32")),
                LabelledSpansOutputs.ENDS: Sequence(Value("int32")),
                LabelledSpansOutputs.LABELS: Sequence(labels_feature),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, labels_feature, spans):
        # pack features together
        spans_begin, spans_end, spans_label = zip(*spans)
        # convert labels to label ids
        if isinstance(labels_feature, ClassLabel):
            spans_label = labels_feature.str2int(spans_label)
        # return all features new
        return {
            LabelledSpansOutputs.BEGINS: [list(spans_begin)],
            LabelledSpansOutputs.ENDS: [list(spans_end)],
            LabelledSpansOutputs.LABELS: [list(spans_label)],
        }


class TestTokenSpansFromBioTagsWithCostumOutputFormat(
    TestTokenSpansFromBioTags
):
    @pytest.fixture
    def processor(self):
        return TokenSpansFromBioTags(
            TokenSpansFromBioTagsConfig(
                bio_tags="bio_tags",
                output_format={
                    LabelledSpansOutputs.BEGINS: LabelledSpansOutputs.BEGINS,
                    LabelledSpansOutputs.ENDS: LabelledSpansOutputs.ENDS,
                    "custom_output_labels_column": LabelledSpansOutputs.LABELS,
                },
            )
        )

    @pytest.fixture
    def expected_out_feature(self, labels_feature):
        return Features(
            {
                LabelledSpansOutputs.BEGINS: Sequence(Value("int32")),
                LabelledSpansOutputs.ENDS: Sequence(Value("int32")),
                "custom_output_labels_column": Sequence(labels_feature),
            }
        )

    @pytest.fixture
    def expected_out_batch(self, labels_feature, spans):
        # pack features together
        spans_begin, spans_end, spans_label = zip(*spans)
        # convert labels to label ids
        if isinstance(labels_feature, ClassLabel):
            spans_label = labels_feature.str2int(spans_label)
        # return all features new
        return {
            LabelledSpansOutputs.BEGINS: [list(spans_begin)],
            LabelledSpansOutputs.ENDS: [list(spans_end)],
            "custom_output_labels_column": [list(spans_label)],
        }


class TestTokenSpansFromBioTagsInvalidTagSequence(TestTokenSpansFromBioTags):
    @pytest.fixture(
        params=[
            # sequence starts with an in tag
            [(0, 7, "X")],
            [(0, 7, "X"), (9, 12, "Y"), (15, 27, "Y")],
            # new entity starts with in tag, i.e. [..., O, I-, ...]
            [(4, 7, "X")],
            [(4, 7, "X"), (9, 12, "Y"), (15, 27, "Y")],
            [(4, 7, "Y"), (9, 12, "X"), (15, 27, "Y")],
            [(4, 7, "Y"), (9, 12, "Y"), (15, 27, "X")],
            # new entity starts with in tag, i.e. [..., I-Y, I-X, ...]
            [(3, 7, "Y"), (7, 12, "X")],
            [(0, 2, "Y"), (3, 7, "Y"), (7, 12, "X")],
        ]
    )
    def spans(self, request):
        return request.param

    @pytest.fixture
    def in_batch(self, spans, tags_feature):
        tags = ["O"] * 32
        # add entity spans
        for b, e, label in spans:
            # add all in tags spanning the whole entity
            # including the very first
            tags[b:e] = ["I-%s" % label] * (e - b)
            # overwrite initial in tag with begin tag
            # but only for entity Y, i.e. X is invalid
            if label == "Y":
                tags[b] = "B-%s" % label

        # convert to ids if needed
        if isinstance(tags_feature, ClassLabel):
            tags = tags_feature.str2int(tags)

        return {"bio_tags": [tags]}

    @pytest.fixture
    def expected_err_on_process(self):
        return ValueError
