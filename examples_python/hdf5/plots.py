import pathlib
import typing
import seaborn
import pandas as pd
import math
import numpy as np
from matplotlib import pyplot


import reports


def _violin_lfd(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
) -> pyplot.Axes:
    ax.violinplot(
        dataset=[[c.lfd for c in clusters] for clusters in clusters_by_depth],
        positions=list(range(len(clusters_by_depth))),
    )
    return ax

def insert_empty_bins(bins: np.ndarray, grouped_data: pd.Series):
    values = list(grouped_data.values)
    for i in range(len(bins)):
        if round(bins[i], 3) not in [key.left for key in grouped_data.keys()]:
            values.insert(i, 0)
    return values


def _heat_lfd(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    normalization_type: typing.Literal["cluster", "cardinality"] = "cardinality",
    num_lfd_buckets: int = 50,
    depth_bucket_step: int = 2,
) -> pyplot.Axes:

    # Regroups clusters_by_depth based on desired size of depth buckets and
    # replaes each cluster with its lfd s.t. the ith sublist in bucketed_lfds_by_depth
    # contains lfds of all clusters whose depth is between i*depth_bucket_step
    # and (i+1)*depth_bucket_step
    bucketed_lfds_by_depth = [
        [
            cluster.lfd
            for depth in clusters_by_depth
            for cluster in depth
            if clusters_by_depth.index(depth) // depth_bucket_step == d
        ]
        for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
    ]

    if normalization_type == "cardinality":
        bucketed_cardinalities_by_depth = [
            [
                cluster.cardinality
                for depth in clusters_by_depth
                for cluster in depth
                if clusters_by_depth.index(depth) // depth_bucket_step == d
            ]
            for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
        ]

    # Creates list of bins for lfd based on desired number of lfd buckets and the max
    # lfd across all clusters in the tree
    max_lfd = max([lfd for depth in bucketed_lfds_by_depth for lfd in depth])
    bins = np.arange(0, max_lfd + max_lfd / num_lfd_buckets, max_lfd / num_lfd_buckets)

    # .cut() segments each sublist in bucketed_lfds_by_depth by lfd bucket and value_counts()
    # counts how many clusters within a particular depth sublist fall into each lfd category
    # Thse lists need to be reversed with [::-1] to correct the fact that seaborn likes to invert
    # the y axes in heatmaps.
    # The array also needs to be transposed in order to get lfd on the y axis.
    if normalization_type == "cluster":
        data = np.transpose(
            [
                (pd.cut(lfds_set, bins=bins).value_counts() / len(lfds_set))[
                    ::-1
                ].to_list()
                for lfds_set in bucketed_lfds_by_depth
            ]
        )
    else:
        data = np.transpose(
            [
                insert_empty_bins(
                    bins,
                    pd.Series(cards_set, index=pd.cut(lfds_set, bins).to_list())
                    .groupby(level=0)
                    .sum()
                    / sum(cards_set),
                )[::-1]
                for lfds_set, cards_set in zip(
                    bucketed_lfds_by_depth, bucketed_cardinalities_by_depth
                )
            ]
        )

    # vmax currently clamped to 0.5. May cause issues with datasets whose lfd is more consistent
    # across clusters in the same depth. Setting to 1 makes it difficult to see color variation; allowing
    # it to be automatic makes it difficult to compare heatmaps for different datasets.
    return seaborn.heatmap(
        data=data,
        vmin=0,
        vmax=0.5,
        cmap="Reds",
        annot=False,
        linewidths=0.75,
        linecolor="white",
        ax=ax,
    )


def _set_violin_lfd_labels(
    clusters_by_depth: list[list[reports.ClusterReport]], ax: pyplot.Axes
):
    ax.set_xlabel("depth - num_clusters")
    ax.set_ylabel("local fractal dimension")

    ax.set_xticks(
        [d for d in range(len(clusters_by_depth))],
        [f"{d}-{len(clusters)}" for d, clusters in enumerate(clusters_by_depth)],
        rotation=270,
    )

    return


def _set_heat_lfd_labels(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    num_lfd_buckets: int = 50,
    depth_bucket_step: int = 2,
):
    ax.set_xlabel("depth")
    ax.set_ylabel("local fractal dimension")

    ax.set_xticks(
        [d for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))],
        [
            f"{depth_bucket_step*d}"
            for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
        ],
        rotation=270,
    )

    max_lfd = max([cluster.lfd for depth in clusters_by_depth for cluster in depth])
    lfd_step = round(max_lfd / num_lfd_buckets, 1)

    ax.set_yticks(
        [d for d in range(num_lfd_buckets)],
        [f"{round(lfd_step*d, 2)}" for d in range(num_lfd_buckets, 0, -1)],
    )

    return


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

    ax = (_violin_lfd if mode == "violin" else _heat_lfd)(
        clusters_by_depth,
        pyplot.axes((0.05, 0.1, 0.9, 0.85)),
    )

    (_set_violin_lfd_labels if mode == "violin" else _set_heat_lfd_labels)(
        clusters_by_depth, ax
    )

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(f"{mode}-lfd-{tree.data_name}__{tree.metric_name}.png"),
            dpi=300,
        )
    pyplot.close(figure)

    return


def _violin_ratio(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    ratio_index: int,
):
    ax.violinplot(
        dataset=[
            [c.ratios[ratio_index] for c in clusters] for clusters in clusters_by_depth
        ],
        positions=list(range(len(clusters_by_depth))),
    )
    return ax


def _heat_ratio(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    ratio_index: int,
    normalization_type: typing.Literal["cluster", "cardinality"] = "cardinality",
    num_ratio_buckets: int = 50,
    depth_bucket_step: int = 2,
):

    bucketed_ratios_by_depth = [
        [
            cluster.ratios[ratio_index]
            for depth in clusters_by_depth
            for cluster in depth
            if clusters_by_depth.index(depth) // depth_bucket_step == d
        ]
        for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
    ]

    if normalization_type == "cardinality":
        bucketed_cardinalities_by_depth = [
            [
                cluster.cardinality
                for depth in clusters_by_depth
                for cluster in depth
                if clusters_by_depth.index(depth) // depth_bucket_step == d
            ]
            for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
        ]

    max_ratio = max([ratio for depth in bucketed_ratios_by_depth for ratio in depth])
    bins = np.arange(
        0, max_ratio + max_ratio / num_ratio_buckets, max_ratio / num_ratio_buckets
    )

    if normalization_type == "cluster":
        data = np.transpose(
            [
                (pd.cut(ratios_set, bins=bins).value_counts() / len(ratios_set))[
                    ::-1
                ].to_list()
                for ratios_set in bucketed_ratios_by_depth
            ]
        )
    else:
        data = np.transpose(
            [
                insert_empty_bins(
                    bins,
                    pd.Series(cards_set, index=pd.cut(ratios_set, bins).to_list())
                    .groupby(level=0)
                    .sum()
                    / sum(cards_set),
                )[::-1]
                for ratios_set, cards_set in zip(
                    bucketed_ratios_by_depth, bucketed_cardinalities_by_depth
                )
            ]
        )

    return seaborn.heatmap(
        data=data,
        vmin=0,
        vmax=0.5,
        cmap="Reds",
        annot=False,
        linewidths=0.75,
        linecolor="white",
        ax=ax,
    )


def _set_violin_ratio_labels(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    ratio_name: typing.Literal[
        "cardinality", "radius", "lfd", "cardinality_ema", "radius_ema", "lfd_ema"
    ],
):
    ax.set_xlabel("depth - num_clusters")
    ax.set_ylabel(f"{ratio_name}")

    ax.set_xticks(
        [d for d in range(len(clusters_by_depth))],
        [f"{d}-{len(clusters)}" for d, clusters in enumerate(clusters_by_depth)],
        rotation=270,
    )

    return


def _set_heat_ratio_labels(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    ratio_name: typing.Literal[
        "cardinality", "radius", "lfd", "cardinality_ema", "radius_ema", "lfd_ema"
    ],
    ratio_index: int,
    num_lfd_buckets: int = 50,
    depth_bucket_step: int = 2,
):
    ax.set_xlabel("depth")
    ax.set_ylabel(f"{ratio_name}")

    ax.set_xticks(
        [d for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))],
        [
            f"{depth_bucket_step*d}"
            for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
        ],
        rotation=270,
    )

    max_ratio = max(
        [
            cluster.ratios[ratio_index]
            for depth in clusters_by_depth
            for cluster in depth
        ]
    )
    lfd_step = round(max_ratio / num_lfd_buckets, 1)

    ax.set_yticks(
        [d for d in range(num_lfd_buckets)],
        [f"{round(lfd_step*d, 2)}" for d in range(num_lfd_buckets, 0, -1)],
    )

    return


def plot_ratios_vs_depth(
    mode: typing.Literal["violin", "heat"],
    tree: reports.TreeReport,
    clusters_by_depth: list[list[reports.ClusterReport]],
    show: bool,
    output_dir: pathlib.Path,
    ratio_name: typing.Literal[
        "cardinality", "radius", "lfd", "cardinality_ema", "radius_ema", "lfd_ema"
    ],
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

    ratios = {
        "cardinality": 0,
        "radius": 1,
        "lfd": 2,
        "cardinality_ema": 3,
        "radius_ema": 4,
        "lfd_ema": 5,
    }

    ax = (_violin_ratio if mode == "violin" else _heat_ratio)(
        clusters_by_depth, pyplot.axes((0.05, 0.1, 0.9, 0.85)), ratios[ratio_name]
    )

    if mode == "violin":
        _set_violin_ratio_labels(
            clusters_by_depth,
            ax,
            ratio_name,
        )
    else:
        _set_heat_ratio_labels(
            clusters_by_depth, ax, ratio_name, ratios[ratio_name], 50, 2
        )

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(
                f"{mode}-{ratio_name}-{tree.data_name}__{tree.metric_name}.png"
            ),
            dpi=300,
        )
    pyplot.close(figure)

    return
