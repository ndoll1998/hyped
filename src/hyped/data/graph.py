import re
from enum import StrEnum
from itertools import count, groupby

import networkx as nx
import numpy as np
from datasets import Features
from matplotlib import colormaps
from matplotlib import pyplot as plt

from hyped.data.pipe import DataPipe
from hyped.data.processors.statistics.base import BaseDataStatistic


class NodeType(StrEnum):
    """Enumeration of node types of a `ProcessGraph`"""

    INPUT_FEATURE = "input_feature"
    OUTPUT_FEATURE = "output_feature"
    DATA_PROCESSOR = "data_processor"
    DATA_STATISTIC = "data_statistic"


class NodeAttribute(StrEnum):
    """Enumeration of node attributes"""

    TYPE = "node_type"
    """The type of the node. Values to this attribute are defined by the
    `NodeType` enumeration."""

    LABEL = "node_label"
    """The label of the node, for features this refers to the feature name,
    for processors it is the processor class name"""

    LAYER = "node_layer"
    """The layer of the node, indicating the path length from the input
    features at layer 0 to the given node"""

    EXECUTION_INDEX = "node_execution_index"
    """The execution index of the node, for processors this refers to the
    order in which processors are executed in the pipeline. For feature
    nodes, the execution index is undefined (set to -1)."""


class EdgeAttribute(StrEnum):
    """Enumeration of edge attributes"""

    FEATURES = "edge_features"
    """A list of feature keys refering to the features that are passed from
    the source node of the edge to the target node."""


class ProcessGraph(nx.DiGraph):
    """Process Graph

    Graph representation of a `DataPipe` object. It consists of all input and
    output features as well as the processors that process the input to the
    output features. Edges show the information flow between processors.

    Nodes and edges have a set of attributes. See `NodeAttributes` and
    `EdgeAttributes` for more details.

    Arguments:
        features (Features): input features to be processed by the data pipe
        pipe (DataPipe): data pipe to represent as a directed graph
    """

    def __init__(self, features: Features, pipe: DataPipe) -> None:
        # create an empty graph
        super(ProcessGraph, self).__init__()
        self.build_process_graph(features, pipe)

    def build_process_graph(self, features: Features, pipe: DataPipe) -> None:
        """Build the graph given the input features and data pipe

        Arguments:
            features (Features):
                input features to be processed by the data pipe
            pipe (DataPipe): data pipe to represent as a directed graph
        """
        # prepare pipeline
        pipe.prepare(features)

        counter = count()
        # create source nodes from input features
        nodes = {k: next(counter) for k in features.keys()}
        layers = {k: 0 for k in features.keys()}
        self.add_nodes_from(
            [
                (
                    i,
                    {
                        NodeAttribute.TYPE: NodeType.INPUT_FEATURE,
                        NodeAttribute.LABEL: k,
                        NodeAttribute.LAYER: layers[k],
                        NodeAttribute.EXECUTION_INDEX: -1,
                    },
                )
                for k, i in nodes.items()
            ]
        )

        # add all processor nodes
        for i, p in enumerate(pipe, 0):
            req_keys = [
                k[0] if isinstance(k, tuple) else k
                for k in p.required_feature_keys
            ]

            # create node attributes
            node_id = next(counter)
            layer = max((layers[k] for k in req_keys), default=0) + 1
            # add node to graph
            self.add_node(
                node_id,
                **{
                    NodeAttribute.TYPE: (
                        NodeType.DATA_STATISTIC
                        if isinstance(p, BaseDataStatistic)
                        else NodeType.DATA_PROCESSOR
                    ),
                    NodeAttribute.LABEL: type(p).__name__,
                    NodeAttribute.LAYER: layer,
                    NodeAttribute.EXECUTION_INDEX: i,
                },
            )

            # group required inputs by the node that provides them
            group = sorted(req_keys, key=lambda k: nodes[k])
            group = groupby(group, key=lambda k: nodes[k])
            # add incoming edges
            for src_node_id, keys in group:
                self.add_edge(
                    src_node_id,
                    node_id,
                    **{EdgeAttribute.FEATURES: list(keys)},
                )

            # update current features
            if not p.config.keep_input_features:
                nodes.clear()
            for k in p.new_features.keys():
                nodes[k] = node_id
                layers[k] = layer

        # add output feature nodes
        for k, src_node_id in nodes.items():
            node_id = next(counter)
            self.add_node(
                node_id,
                **{
                    NodeAttribute.TYPE: NodeType.OUTPUT_FEATURE,
                    NodeAttribute.LABEL: k,
                    NodeAttribute.LAYER: layers[k] + 1,
                    NodeAttribute.EXECUTION_INDEX: -1,
                },
            )
            self.add_edge(
                src_node_id, node_id, **{EdgeAttribute.FEATURES: [k]}
            )

    @property
    def num_layers(self) -> int:
        """The total number of layers in the graph.

        It is computed by finding the maximum value of the
        `NodeAttribute.LAYER` attributes of the nodes.
        """
        return (
            max(nx.get_node_attributes(self, NodeAttribute.LAYER).values()) + 1
        )

    @property
    def max_width(self) -> int:
        """The maximum number of nodes contained in a single layer."""
        # group nodes by their layer
        layers = nx.get_node_attributes(self, NodeAttribute.LAYER)
        layers = groupby(sorted(self, key=layers.get), key=layers.get)
        # find larges layer in graph
        return max(len(list(layer)) for _, layer in layers)

    def plot(
        self,
        pos: None | dict[int, list[float]] = None,
        color_map: dict[NodeType, str] = {},
        with_labels: bool = True,
        with_edge_labels: bool = True,
        font_size: int = 10,
        node_size: int = 10_000,
        arrowsize: int = 25,
        ax: None | plt.Axes = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot the graph

        Arguments:
            pos (None | dict[str, list[float]]):
                positions of the nodes, when set to None (default) the node
                positions are computed using `networkx.multipartite_layout`
                based on the `NodeAttribute.LAYER` attribute of the nodes.
            color_map (dict[NodeType, str]):
                indicate custom color scheme based on node type
            with_labels (bool): plot node labels
            with_edge_labels (bool): plot edge labels
            font_size (int): font size used for labels
            node_size (int): size of nodes
            arrowsize (int): size of arrows
            ax (None | plt.Axes): axes to plot in
            **kwargs: arguments forwarded to networkx.draw

        Returns:
            ax (plt.Axes): axes containing plot of the graph
        """

        # limit the maximum number of character in a single line in nodes
        max_node_line_length = node_size // (font_size * 65)

        cmap = colormaps.get_cmap("Pastel1")
        # build full color map
        default_color_map = {
            NodeType.INPUT_FEATURE: cmap.colors[0],
            NodeType.OUTPUT_FEATURE: cmap.colors[0],
            NodeType.DATA_PROCESSOR: cmap.colors[1],
            NodeType.DATA_STATISTIC: cmap.colors[2],
        }
        color_map = default_color_map | color_map

        # get node and edge attributes
        node_types = nx.get_node_attributes(self, NodeAttribute.TYPE)
        node_index = nx.get_node_attributes(
            self, NodeAttribute.EXECUTION_INDEX
        )
        node_labels = nx.get_node_attributes(self, NodeAttribute.LABEL)
        edge_features = nx.get_edge_attributes(self, EdgeAttribute.FEATURES)
        # convert node types to colors
        node_color = [color_map.get(node_types[node], "red") for node in self]

        def add_line_breaks(mixed_case_string: str, max_line_length: int):
            # split string into words
            words = re.split(r"(?<=[a-z])(?=[A-Z])", mixed_case_string)
            # group words such that each group has a limited number of
            # characers
            lengths = np.cumsum(list(map(len, words))) // max_line_length
            groups = groupby(range(len(words)), key=lengths.__getitem__)
            # join groups with newlines inbetween
            return "\n".join(
                ["".join([words[i] for i in group]) for _, group in groups]
            )

        def get_node_label(node):
            if node_types[node] in (
                NodeType.DATA_PROCESSOR,
                NodeType.DATA_STATISTIC,
            ):
                formatted_label = add_line_breaks(
                    node_labels[node], max_node_line_length
                )
                return "[%i]\n%s" % (node_index[node], formatted_label)
            if node_types[node] in (
                NodeType.INPUT_FEATURE,
                NodeType.OUTPUT_FEATURE,
            ):
                return "[%s]" % node_labels[node]

        def get_edge_label(edge):
            # TODO: create string representation of complex keys
            return "\n".join(edge_features[edge])

        # create a plot axes
        if ax is None:
            _, ax = plt.subplots(
                1, 1, figsize=(self.num_layers * 5, self.max_width * 5)
            )

        # compute node positions when not provided
        pos = (
            pos
            if pos is not None
            else nx.multipartite_layout(self, subset_key=NodeAttribute.LAYER)
        )

        # build display labels for nodes
        node_display_labels = {
            node: get_node_label(node) for node in self.nodes
        }
        # draw graph
        nx.draw(
            self,
            pos,
            with_labels=with_labels,
            labels=node_display_labels,
            node_color=node_color,
            node_size=node_size,
            font_size=font_size,
            arrowsize=arrowsize,
            ax=ax,
            **kwargs,
        )

        # add edge labels
        if with_edge_labels:
            edge_display_labels = {
                edge: get_edge_label(edge) for edge in self.edges
            }
            nx.draw_networkx_edge_labels(
                self,
                pos,
                edge_labels=edge_display_labels,
                ax=ax,
                font_size=font_size,
            )

        return ax
