from .outputs import LabelledSpansOutputs
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)
from hyped.utils.feature_checks import (
    raise_feature_is_sequence,
    get_sequence_feature,
)
from hyped.utils.feature_access import (
    FeatureKey,
    get_feature_at_key,
    get_value_at_key,
)
from datasets import Features, Sequence, Value, ClassLabel
from dataclasses import dataclass
from typing import Literal, Any


@dataclass
class TokenSpansFromBioTagsConfig(BaseDataProcessorConfig):
    """Token Spans from Begin-In-Out (BIO) Tags Processor Config

    Convert Bio Tags to token-level spans

    Attributes:
        begin_tag_prefix (str): tag prefix marking begin tags
        in_tag_prefix (str): tag prefix marking in tags
        out_tag (str): out tag
        bio_tags (FeatureKey):
            input feature containing the bio tag sequence to parse
    """

    t: Literal[
        "hyped.data.processors.spans.from_bio"
    ] = "hyped.data.processors.spans.from_bio"

    # tag schema
    begin_tag_prefix: str = "B-"
    in_tag_prefix: str = "I-"
    out_tag: str = "O"
    # bio tags
    bio_tags: FeatureKey = None


class TokenSpansFromBioTags(BaseDataProcessor[TokenSpansFromBioTagsConfig]):
    """Token Spans from Begin-In-Out (BIO) Tags Processor

    Convert Bio Tags to token-level spans
    """

    @property
    def tag_feature(self) -> Value | ClassLabel:
        """The item feature of the tag sequence"""
        feature = get_feature_at_key(self.in_features, self.config.bio_tags)
        return get_sequence_feature(feature)

    @property
    def label_feature(self) -> Value | ClassLabel:
        """The item feature of the label sequence"""
        feature = self.raw_features[LabelledSpansOutputs.LABELS]
        return get_sequence_feature(feature)

    def map_features(self, features: Features) -> Features:
        """Check input features and return feature mapping
        for token-level span annotations and labels.

        If the bio tags feature is a sequence of class labels,
        then the entity labels are extracted from the BIO label
        scheme. Otherwise the output feature will be a sequence
        of strings indicating the entity labels.

        Arguments:
            features (Features): input dataset features

        Returns:
            out (Features): token-level span annotation features
        """
        # make sure bio tags feature exists and is a sequence
        # of either class labels or strings indicating the label
        feature = get_feature_at_key(features, self.config.bio_tags)
        raise_feature_is_sequence(
            self.config.bio_tags,
            feature,
            (Value("string"), ClassLabel),
        )
        # get the item feature
        feature = features[self.config.bio_tags]
        feature = get_sequence_feature(feature)

        if isinstance(feature, ClassLabel):
            # remove prefix from the class labels to get the entity
            # labels to each tag, note that this list contains dulpicates
            # as to each label there is both a begin and in tag
            names = [
                (
                    name.removeprefix(
                        self.config.begin_tag_prefix
                    ).removeprefix(self.config.in_tag_prefix)
                )
                for name in feature.names
                if name != self.config.out_tag
            ]
            # make names unqiue while preserving order
            names = list(dict.fromkeys(names))
            # create label sequence feature
            return {
                LabelledSpansOutputs.BEGINS.value: Sequence(Value("int32")),
                LabelledSpansOutputs.ENDS.value: Sequence(Value("int32")),
                LabelledSpansOutputs.LABELS.value: Sequence(
                    ClassLabel(names=names)
                ),
            }

        return {
            LabelledSpansOutputs.BEGINS.value: Sequence(Value("int32")),
            LabelledSpansOutputs.ENDS.value: Sequence(Value("int32")),
            LabelledSpansOutputs.LABELS.value: Sequence(Value("string")),
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
            out (dict[str, Any]): token-level span annotations
        """

        tags = get_value_at_key(example, self.config.bio_tags)
        # convert tag-ids to tags
        if isinstance(self.tag_feature, ClassLabel):
            tags = self.tag_feature.int2str(tags)

        # convert tags to labels
        labels = [
            (
                tag.removeprefix(self.config.begin_tag_prefix).removeprefix(
                    self.config.in_tag_prefix
                )
            )
            for tag in tags
        ]

        # features to extract
        spans_begin = []
        spans_end = []
        spans_label = []

        for i, (tag, label) in enumerate(zip(tags, labels)):
            if tag == self.config.out_tag:
                continue

            if tag.startswith(self.config.begin_tag_prefix):
                spans_begin.append(i)
                spans_end.append(i + 1)
                spans_label.append(label)

            if tag.startswith(self.config.in_tag_prefix):
                # sequence cannot start with an in-tag
                if i == 0:
                    raise ValueError(
                        "Sequence doesn't conform to BIO labeling scheme: "
                        "Sequence starts with an In-tag: %s" % str(tags[:10])
                    )
                # entity cannot start with an in-tag
                if (tags[i - 1] == self.config.out_tag) or (
                    label != labels[i - 1]
                ):
                    raise ValueError(
                        "Sequence doesn't conform to BIO labeling scheme: "
                        "New Entity starts with an In-tag: %s"
                        % str(tags[max(i - 4, 0) : i + 4])  # noqa: E203
                    )

                assert label == spans_label[-1]
                # update end of the span
                spans_end[-1] = i + 1

        # convert labels to label ids if needed
        if isinstance(self.label_feature, ClassLabel):
            spans_label = self.label_feature.str2int(spans_label)

        # some sanity checking
        assert len(spans_begin) == len(spans_end) == len(spans_label)

        return {
            LabelledSpansOutputs.BEGINS.value: spans_begin,
            LabelledSpansOutputs.ENDS.value: spans_end,
            LabelledSpansOutputs.LABELS.value: spans_label,
        }
