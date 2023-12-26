import networkx as nx
from datasets import Features
from enum import StrEnum
from matplotlib import pyplot as plt
from itertools import count, groupby
from .pipe import DataPipe


class NodeType(StrEnum):
    """Enumeration of node types of a `ProcessGraph`"""

    INPUT_FEATURE = "input_feature"
    OUTPUT_FEATURE = "output_feature"
    DATA_PROCESSOR = "data_processor"


# TODO: test graph construction
class ProcessGraph(nx.DiGraph):
    """Process Graph

    Graph representation of a `DataPipe` object. It consists of all input and
    output features as well as the processors that process the input to the
    output features. Edges show the information flow between processors.

    For valid node types see the `NodeType` enumeration.

    Node Arguments:
        label (str):
            the label of the node, for features this is the feature name,
            for processors it is the processor class name
        type (NodeType): the type of the node
        layer (int):
            layer of the node, indicating the path length from the input
            features to the given node

    Edge Attributes:
        label (str):
            comma separated list of all features passed from the source
            to the target node of the edge

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
                        "label": "[%s]" % k,
                        "type": NodeType.INPUT_FEATURE,
                        "layer": layers[k],
                    },
                )
                for k, i in nodes.items()
            ]
        )

        # add all processor nodes
        for p in pipe:
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
                label=type(p).__name__,
                type=NodeType.DATA_PROCESSOR,
                layer=layer,
            )

            # group required inputs by the node that provides them
            group = sorted(req_keys, key=lambda k: nodes[k])
            group = groupby(group, key=lambda k: nodes[k])
            # add incoming edges
            for src_node_id, keys in group:
                self.add_edge(src_node_id, node_id, label="\n".join(keys))

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
                label="[%s]" % k,
                type=NodeType.OUTPUT_FEATURE,
                layer=layers[k] + 1,
            )
            # TODO: create string representation of complex keys
            self.add_edge(src_node_id, node_id, label=k)

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
                based on the `layer` attribute of the nodes.
            color_map (dict[NodeType, str]):
                indicate custom color scheme based on node type
            with_labels (bool): plot node labels
            with_edge_labels (bool): plot edge labels
            font_size (int): font size used for labels
            node_size (int): size of nodes
            arrowsize (int): size of arrows
            ax (plt.Axes): axes to plot in
            **kwargs: arguments forwarded to networkx.draw
        """

        # build full color map
        default_color_map = {
            NodeType.INPUT_FEATURE: "lightpink",
            NodeType.OUTPUT_FEATURE: "lightpink",
            NodeType.DATA_PROCESSOR: "lightblue",
        }
        color_map = default_color_map | color_map

        # get node and edge attributes
        node_labels = nx.get_node_attributes(self, "label")
        edge_labels = nx.get_edge_attributes(self, "label")
        node_types = nx.get_node_attributes(self, "type")
        # convert node types to colors
        node_color = [color_map.get(node_types[node], "red") for node in self]

        # create a plot axes
        if ax is None:
            _, ax = plt.subplots(1, 1)

        # compute node positions when not provided
        pos = (
            pos
            if pos is not None
            else nx.multipartite_layout(self, subset_key="layer")
        )

        # draw graph
        nx.draw(
            self,
            pos,
            with_labels=with_labels,
            labels=node_labels,
            node_color=node_color,
            node_size=node_size,
            font_size=font_size,
            arrowsize=arrowsize,
            ax=ax,
            **kwargs,
        )
        # add edge labels
        if with_edge_labels:
            nx.draw_networkx_edge_labels(
                self, pos, edge_labels=edge_labels, ax=ax, font_size=font_size
            )

        return ax
