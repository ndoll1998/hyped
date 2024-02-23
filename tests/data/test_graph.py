import warnings
from itertools import chain

import networkx as nx
import pytest
from datasets import ClassLabel, Features, Sequence, Value

from hyped.data.graph import NodeAttribute, NodeType, ProcessGraph
from hyped.data.pipe import DataPipe
from hyped.data.processors.base import BaseDataProcessor
from hyped.data.processors.features.filter import (
    FilterFeatures,
    FilterFeaturesConfig,
)
from hyped.data.processors.features.format import (
    FormatFeatures,
    FormatFeaturesConfig,
)
from hyped.data.processors.spans.apply_idx_spans import (
    ApplyIndexSpans,
    ApplyIndexSpansConfig,
)
from hyped.data.processors.spans.from_bio import (
    TokenSpansFromBioTags,
    TokenSpansFromBioTagsConfig,
)
from hyped.data.processors.spans.from_word_ids import (
    TokenSpansFromWordIds,
    TokenSpansFromWordIdsConfig,
)
from hyped.data.processors.spans.outputs import (
    LabelledSpansOutputs,
    SpansOutputs,
)
from hyped.data.processors.statistics.base import BaseDataStatistic
from hyped.data.processors.statistics.value.mean_and_std import (
    MeanAndStd,
    MeanAndStdConfig,
)
from hyped.data.processors.taggers.bio import (
    BioTagger,
    BioTaggerConfig,
    BioTaggerOutputs,
)
from hyped.data.processors.tokenizers.hf import (
    HuggingFaceTokenizer,
    HuggingFaceTokenizerConfig,
    HuggingFaceTokenizerOutputs,
)


class BaseTestProcessGraph(object):
    @pytest.fixture
    def features(self) -> Features:
        raise NotImplementedError

    @pytest.fixture
    def pipe(self) -> DataPipe:
        raise NotImplementedError

    @pytest.fixture
    def num_layers(self) -> int:
        raise NotImplementedError

    @pytest.fixture
    def max_width(self) -> int:
        raise NotImplementedError

    @pytest.fixture
    def graph(self) -> nx.DiGraph:
        raise NotImplementedError

    @pytest.fixture
    def G(self, features, pipe) -> ProcessGraph:
        # ignore warning that no statistics report is active
        with warnings.catch_warnings(category=UserWarning, action="ignore"):
            return ProcessGraph(features, pipe)

    def test_nodes(self, G, features, pipe):
        # ignore warning that no statistics report is active
        with warnings.catch_warnings(category=UserWarning, action="ignore"):
            # make sure the pipe is prepared with the features
            pipe.prepare(features)

        # get node attributes
        node_types = nx.get_node_attributes(G, NodeAttribute.TYPE)
        node_label = nx.get_node_attributes(G, NodeAttribute.LABEL)
        node_index = nx.get_node_attributes(G, NodeAttribute.EXECUTION_INDEX)

        # separate node types
        processors = [n for n in G if node_types[n] is NodeType.DATA_PROCESSOR]
        statistics = [n for n in G if node_types[n] is NodeType.DATA_STATISTIC]
        in_features = [n for n in G if node_types[n] is NodeType.INPUT_FEATURE]
        out_features = [
            n for n in G if node_types[n] is NodeType.OUTPUT_FEATURE
        ]
        # make sure all data processors are present
        assert len(pipe) == len(processors) + len(statistics)
        # test all processors
        for node in processors:
            idx = node_index[node]
            assert node_label[node] == type(pipe[idx]).__name__  # noqa: E721
            assert isinstance(pipe[idx], BaseDataProcessor)
            assert not isinstance(pipe[idx], BaseDataStatistic)
        # test all statistics
        for node in statistics:
            idx = node_index[node]
            assert node_label[node] == type(pipe[idx]).__name__  # noqa: E721
            assert isinstance(pipe[idx], BaseDataStatistic)

        # make sure all input features are present
        assert len(features) == len(in_features)
        for k in features.keys():
            assert k in {node_label[node] for node in in_features}

        # make sure all output features are present
        assert len(pipe.out_features) == len(out_features)
        for k in pipe.out_features.keys():
            assert k in {node_label[node] for node in out_features}

        # test execution index of feature nodes
        for node in chain(in_features, out_features):
            assert node_index[node] == -1

    def test_topology(self, G, graph, num_layers, max_width):
        # test graph attributes
        assert G.num_layers == num_layers
        assert G.max_width == max_width

        node_layers = nx.get_node_attributes(G, NodeAttribute.LAYER)
        # get all sources
        node_types = nx.get_node_attributes(G, NodeAttribute.TYPE)
        sources = [i for i in G if node_types[i] is NodeType.INPUT_FEATURE]
        # test layer attribute
        for node in G:
            if node in sources:
                # sources are placed at layer 0
                assert node_layers[node] == 0
            else:
                # layer of other nodes is the length of the longest path from
                # any input feature to the node
                paths = chain.from_iterable(
                    nx.all_simple_paths(G, source, node) for source in sources
                )
                max_length = max(map(len, paths), default=0) - 1
                # make sure node is connected to any input feature
                assert max_length != -1
                assert node_layers[node] == max_length

        # test if graph matches expected topology
        assert nx.is_isomorphic(G, graph)


class TestSimplePath(BaseTestProcessGraph):
    @pytest.fixture
    def features(self) -> Features:
        return Features({"x": Value("int32")})

    @pytest.fixture
    def pipe(self) -> DataPipe:
        return DataPipe(
            [
                FormatFeatures(
                    FormatFeaturesConfig(
                        output_format={"y": "x"}, keep_input_features=False
                    )
                ),
                FormatFeatures(
                    FormatFeaturesConfig(
                        output_format={"x": "y"}, keep_input_features=False
                    )
                ),
            ]
        )

    @pytest.fixture
    def num_layers(self) -> int:
        return 4

    @pytest.fixture
    def max_width(self) -> int:
        return 1

    @pytest.fixture
    def graph(self) -> nx.DiGraph:
        return nx.DiGraph([(0, 1), (1, 2), (2, 3)])


class TestSimpleTree(BaseTestProcessGraph):
    @pytest.fixture
    def features(self) -> Features:
        return Features({"x": Value("int32")})

    @pytest.fixture
    def pipe(self) -> DataPipe:
        return DataPipe(
            [
                FormatFeatures(
                    FormatFeaturesConfig(
                        output_format={"y": "x", "z": "x"},
                        keep_input_features=True,
                    )
                ),
                FormatFeatures(
                    FormatFeaturesConfig(
                        output_format={"x": "y", "a": "z"},
                        keep_input_features=True,
                    )
                ),
            ]
        )

    @pytest.fixture
    def num_layers(self) -> int:
        return 4

    @pytest.fixture
    def max_width(self) -> int:
        return 3

    @pytest.fixture
    def graph(self) -> nx.DiGraph:
        return nx.DiGraph([(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6)])


class TestTreeWithStatistic(BaseTestProcessGraph):
    @pytest.fixture
    def features(self) -> Features:
        return Features({"x": Value("int32")})

    @pytest.fixture
    def pipe(self) -> DataPipe:
        return DataPipe(
            [
                FormatFeatures(
                    FormatFeaturesConfig(
                        output_format={"y": "x", "z": "x"},
                        keep_input_features=True,
                    )
                ),
                FormatFeatures(
                    FormatFeaturesConfig(
                        output_format={"x": "y", "a": "z"},
                        keep_input_features=True,
                    )
                ),
                MeanAndStd(
                    MeanAndStdConfig(
                        feature_key="y", statistic_key="y.mean_and_std"
                    )
                ),
            ]
        )

    @pytest.fixture
    def num_layers(self) -> int:
        return 4

    @pytest.fixture
    def max_width(self) -> int:
        return 4

    @pytest.fixture
    def graph(self) -> nx.DiGraph:
        return nx.DiGraph(
            [(0, 1), (1, 2), (1, 3), (1, 4), (4, 5), (4, 6), (1, 7)]
        )


class TestNerGraph(BaseTestProcessGraph):
    @pytest.fixture
    def features(self) -> Features:
        return Features(
            {
                "tokens": Sequence(feature=Value(dtype="string")),
                "ner_tags": Sequence(
                    feature=ClassLabel(
                        names=[
                            "O",
                            "B-PER",
                            "I-PER",
                            "B-ORG",
                            "I-ORG",
                            "B-LOC",
                            "I-LOC",
                            "B-MISC",
                            "I-MISC",
                        ]
                    )
                ),
            }
        )

    @pytest.fixture
    def pipe(self) -> DataPipe:
        return DataPipe(
            [
                # apply tokenizer
                HuggingFaceTokenizer(
                    HuggingFaceTokenizerConfig(
                        tokenizer="bert-base-uncased",
                        text="tokens",
                        is_split_into_words=True,
                        return_word_ids=True,
                        return_special_tokens_mask=True,
                    )
                ),
                # convert word ids to token spans
                TokenSpansFromWordIds(
                    TokenSpansFromWordIdsConfig(
                        word_ids=HuggingFaceTokenizerOutputs.WORD_IDS,
                        mask=HuggingFaceTokenizerOutputs.SPECIAL_TOKENS_MASK,
                        output_format={
                            "word_begins": SpansOutputs.BEGINS,
                            "word_ends": SpansOutputs.ENDS,
                        },
                    )
                ),
                # convert word-level bio tags to word-level spans
                TokenSpansFromBioTags(
                    TokenSpansFromBioTagsConfig(
                        bio_tags="ner_tags",
                        output_format={
                            "word_entity_begins": LabelledSpansOutputs.BEGINS,
                            "word_entity_ends": LabelledSpansOutputs.ENDS,
                            "entity_labels": LabelledSpansOutputs.LABELS,
                        },
                    )
                ),
                # convert word-level spans to token-level spans
                ApplyIndexSpans(
                    ApplyIndexSpansConfig(
                        idx_spans_begin="word_entity_begins",
                        idx_spans_end="word_entity_ends",
                        spans_begin="word_begins",
                        spans_end="word_ends",
                        output_format={
                            "token_entity_begins": SpansOutputs.BEGINS,
                            "token_entity_ends": SpansOutputs.ENDS,
                        },
                    )
                ),
                # create token-level bio tags from token-level spans
                BioTagger(
                    BioTaggerConfig(
                        input_sequence=HuggingFaceTokenizerOutputs.INPUT_IDS,
                        mask=HuggingFaceTokenizerOutputs.SPECIAL_TOKENS_MASK,
                        entity_spans_begin="token_entity_begins",
                        entity_spans_end="token_entity_ends",
                        entity_spans_label="entity_labels",
                        output_format={"labels": BioTaggerOutputs.BIO_TAGS},
                    )
                ),
                # filter all unnecessary/intermediate features
                FilterFeatures(
                    FilterFeaturesConfig(
                        keep=[HuggingFaceTokenizerOutputs.INPUT_IDS, "labels"]
                    )
                ),
            ]
        )

    @pytest.fixture
    def num_layers(self) -> int:
        return 7

    @pytest.fixture
    def max_width(self) -> int:
        return 2

    @pytest.fixture
    def graph(self) -> nx.DiGraph:
        return nx.DiGraph(
            [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (1, 4),
                (1, 5),
                (6, 7),
                (7, 3),
                (7, 4),
                (5, 8),
                (5, 9),
            ]
        )
