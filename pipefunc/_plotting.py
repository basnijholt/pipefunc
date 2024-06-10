from __future__ import annotations

import inspect
import re
import warnings
from typing import TYPE_CHECKING, Any

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from pipefunc._utils import at_least_tuple

if TYPE_CHECKING:
    from pathlib import Path

    import holoviews as hv

    from pipefunc._pipefunc import PipeFunc

_empty = inspect.Parameter.empty
MAX_LABEL_LENGTH = 20


def _get_graph_layout(graph: nx.DiGraph) -> dict:
    try:
        return graphviz_layout(graph, prog="dot")
    except ImportError:
        warnings.warn(
            "Graphviz is not installed. Using spring layout instead.",
            ImportWarning,
            stacklevel=2,
        )
        return nx.spring_layout(graph)


def _trim(s: Any, max_len: int = MAX_LABEL_LENGTH) -> str:
    s = str(s)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def visualize(  # noqa: PLR0912, PLR0915
    graph: nx.DiGraph,
    figsize: tuple[int, int] = (10, 10),
    filename: str | Path | None = None,
    func_node_colors: str | list[str] | None = None,
) -> None:
    """Visualize the pipeline as a directed graph.

    Parameters
    ----------
    graph
        The directed graph representing the pipeline.
    figsize
        The width and height of the figure in inches.
    filename
        The filename to save the figure to.
    func_node_colors
        The color of the nodes.

    """
    import matplotlib.pyplot as plt

    from pipefunc._pipefunc import PipeFunc
    from pipefunc._pipeline import _Bound

    pos = _get_graph_layout(graph)
    arg_nodes = []
    func_nodes = []
    bound_nodes = []
    for node in graph.nodes:
        if isinstance(node, str):
            arg_nodes.append(node)
        elif isinstance(node, PipeFunc):
            func_nodes.append(node)
        else:
            assert isinstance(node, _Bound)
            bound_nodes.append(node)

    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=arg_nodes,
        node_size=4000,
        node_color="lightgreen",
        node_shape="s",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=func_nodes,
        node_size=4000,
        node_color=func_node_colors or "skyblue",
        node_shape="o",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=bound_nodes,
        node_size=4000,
        node_color="red",
        node_shape="h",
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        {node: node for node in arg_nodes},
        font_size=12,
    )

    def func_with_mapspec(func: PipeFunc) -> str:
        s = str(func)
        if not func.mapspec:
            return s
        for spec in func.mapspec.outputs:
            # changes e.g., "func(...) -> y" to "func(...) -> y[i]"
            s = re.sub(rf"\b{spec.name}\b", str(spec), s)
        return s

    nx.draw_networkx_labels(
        graph,
        pos,
        {node: func_with_mapspec(node) for node in func_nodes},
        font_size=12,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        {node: node.name for node in bound_nodes},
        font_size=12,
    )

    nx.draw_networkx_edges(graph, pos, arrows=True, node_size=4000)

    # Add edge labels with function outputs
    outputs = {}
    inputs = {}
    bound = {}
    outputs_mapspec = {}
    inputs_mapspec = {}

    for edge, attrs in graph.edges.items():
        a, b = edge
        if isinstance(a, str):
            assert not isinstance(b, str)  # `b` is PipeFunc
            default_value = b.defaults.get(a, _empty)
            if b.mapspec and a in b.mapspec.input_names:
                spec = next(i for i in b.mapspec.inputs if i.name == a)
                inputs_mapspec[edge] = str(spec)
            elif default_value is not _empty:
                inputs[edge] = f"{a}={_trim(default_value)}"
            else:
                inputs[edge] = a
        elif isinstance(a, PipeFunc):
            output_str = []
            with_mapspec = False
            for name in at_least_tuple(attrs["arg"]):
                if b.mapspec and name in b.mapspec.input_names:
                    with_mapspec = True
                    spec = next(i for i in b.mapspec.inputs if i.name == name)
                    output_str.append(str(spec))
                else:
                    output_str.append(name)
            if with_mapspec:
                outputs_mapspec[edge] = ", ".join(output_str)
            else:
                outputs[edge] = ", ".join(output_str)
        else:
            assert isinstance(a, _Bound)
            bound_value = a.value
            bound[edge] = f"{a.name}={bound_value}"

    for labels, color in [
        (outputs, "skyblue"),
        (outputs_mapspec, "blue"),
        (inputs, "lightgreen"),
        (inputs_mapspec, "green"),
        (bound, "red"),
    ]:
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=labels,
            font_size=12,
            font_color=color,
        )

    plt.axis("off")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def visualize_holoviews(graph: nx.DiGraph, *, show: bool = False) -> hv.Graph | None:
    """Visualize the pipeline as a directed graph using HoloViews.

    Parameters
    ----------
    graph
        The directed graph representing the pipeline.
    show
        Whether to show the plot. Uses `bokeh.plotting.show(holoviews.render(plot))`.
        If ``False`` the `holoviews.Graph` object is returned.

    """
    import bokeh.plotting
    import holoviews as hv
    import numpy as np

    from pipefunc._pipefunc import PipeFunc
    from pipefunc._pipeline import _Bound

    hv.extension("bokeh")
    pos = _get_graph_layout(graph)

    # Extract node positions and create a list of nodes
    x, y = np.array([pos[node] for node in graph.nodes]).T
    node_indices = range(len(graph.nodes))
    node_types = []
    node_labels = []
    mapspecs = []
    for node in graph.nodes:
        if isinstance(node, str):
            node_types.append("str")
            node_labels.append(node)
            mapspecs.append(None)
        elif isinstance(node, PipeFunc):
            node_types.append("func")
            node_labels.append(str(node))
            mapspecs.append(node.mapspec)
        else:
            assert isinstance(node, _Bound)
            node_types.append("bound")
            node_labels.append(node.name)
            mapspecs.append(None)

    # Create a dictionary for quick lookup of indices
    node_index_dict = {node: index for index, node in enumerate(graph.nodes)}

    # Extract edge info using the lookup dictionary
    edges = np.array(
        [(node_index_dict[edge[0]], node_index_dict[edge[1]]) for edge in graph.edges],
    )

    # Create Nodes and Graph
    nodes = hv.Nodes(
        (x, y, node_indices, node_labels, node_types, mapspecs),
        vdims=["label", "type", "mapspec"],
    )
    graph = hv.Graph((edges, nodes))

    plot_opts = {
        "width": 800,
        "height": 800,
        "padding": 0.1,
        "xaxis": None,
        "directed": True,
        "node_size": 50,
        "yaxis": None,
        "node_color": hv.dim("type").categorize(
            {
                "str": "lightgreen",
                "func": "skyblue",
                "bound": "red",
            },
            "gray",
        ),
        "edge_color": "black",
    }

    graph = graph.opts(**plot_opts)

    # Create Labels and add them to the graph
    labels = hv.Labels(graph.nodes, ["x", "y"], "label")
    plot = graph * labels.opts(
        text_font_size="8pt",
        text_color="black",
        bgcolor="white",
    )
    if show:
        bokeh.plotting.show(hv.render(plot))
        return None
    return plot
