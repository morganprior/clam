import pathlib
import typing
import seaborn
import pandas as pd
import math
from matplotlib import pyplot


import reports


def _violin(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
) -> pyplot.Axes:
    ax.violinplot(
        dataset=[[c.lfd for c in clusters] for clusters in clusters_by_depth],
        positions=list(range(len(clusters_by_depth))),
    )
    return ax


def _heat(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
) -> pyplot.Axes:
    # 2D array in which each element corresponds to a cluster and is of the form [lfd, depth]
    # lfd grouped by nearest integer; depths bucketed into groups of 5
    bucketed_clusters = pd.DataFrame(
        data=[
            [cluster.lfd // 1, clusters_by_depth.index(depth) // 5]
            for depth in clusters_by_depth
            for cluster in depth
        ],
        columns=["lfd", "depth"],
    )

    cross_tabulation = pd.crosstab(
        bucketed_clusters["lfd"], bucketed_clusters["depth"], dropna=False
    ).div(len(bucketed_clusters))

    return seaborn.heatmap(data=cross_tabulation, cmap="Reds", annot=True, ax=ax)


def _set_violin_labels(
    clusters_by_depth: list[list[reports.ClusterReport]], ax: pyplot.Axes
):
    ax.set_xlabel("depth - num_clusters")
    ax.set_ylabel("local fractal dimension")

    ax.set_xticks(
        [d for d in range(len(clusters_by_depth))],
        [f"{d}-{len(clusters)}" for d, clusters in enumerate(clusters_by_depth)],
        rotation=270,
    )


def _set_heat_labels(
    clusters_by_depth: list[list[reports.ClusterReport]], ax: pyplot.Axes
):
    ax.set_xlabel("depth")
    ax.set_ylabel("local fractal dimension")

    ax.set_xticks(
        [d + 0.5 for d in range(math.ceil(len(clusters_by_depth) / 5))],
        [f"{5*d}-{5*(d+1)}" for d in range(math.ceil(len(clusters_by_depth) / 5))],
        rotation=270,
    )


def plot_lfd_vs_depth(
    mode: typing.Literal["violin", "heat"],
    tree: reports.TreeReport,
    clusters_by_depth: list[list[reports.ClusterReport]],
    show: bool,
    output_dir: pathlib.Path,
):
    figure: pyplot.Figure = pyplot.figure(figsize=(16, 10), dpi=300)
    title = ", ".join(
        [
            f"name = {tree.data_name}",
            f"metric = {tree.metric_name}",
            f"cardinality = {tree.cardinality}",
            f"dimensionality = {tree.dimensionality}",
            f"build_time = {tree.build_time:3.2e} (sec)",
        ]
    )
    figure.suptitle(title)

    ax = (_violin if mode == "violin" else _heat)(
        clusters_by_depth,
        pyplot.axes((0.05, 0.1, 0.9, 0.85)),
    )

    (_set_violin_labels if mode == "violin" else _set_heat_labels)(
        clusters_by_depth, ax
    )

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(f"{mode}-{tree.data_name}__{tree.metric_name}.png"),
            dpi=300,
        )
    pyplot.close(figure)

    return
