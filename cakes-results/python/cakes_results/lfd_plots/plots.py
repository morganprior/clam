"""Create plots for the trees."""

import logging
import pathlib
import shutil

import matplotlib.pyplot as plt
import pandas

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


def create_plots(
    tree_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    """Create plots for clusters in the csv file.

    Args:
        tree_dir: The directory containing the csv file.
        output_dir: The directory to save the plots.
        show: Whether to show the plots instead of saving them.
    """
    # Get the names of the dataset and metric from the csv file name
    tree_name = tree_dir.stem
    dataset, metric = tree_name.split("_")
    logger.info(f"{dataset = }, {metric = }")

    output_dir = output_dir.joinpath(tree_name)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    # The csv file should have the following columns:
    # "offset",
    # "cardinality",
    # "depth",
    # "radius",
    # "lfd",
    # "ratio_cardinality",
    # "ratio_radius",
    # "ratio_lfd",
    # "ratio_cardinality_ema",
    # "ratio_radius_ema",
    # "ratio_lfd_ema",

    # Read the csv file
    df = pandas.read_csv(tree_dir.joinpath("clusters.csv"))

    plotting_properties = [
        "radius",
        "lfd",
        "ratio_cardinality",
        "ratio_radius",
        "ratio_lfd",
        "ratio_cardinality_ema",
        "ratio_radius_ema",
        "ratio_lfd_ema",
    ]

    for prop in plotting_properties:
        logger.info(f"Plotting {prop}")

        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(16, 10))

        # The x-axis will be the depth
        # The y-axis will be the value of the property
        # At each depth, we will draw a violin plot of the property values

        # Group the data by depth
        depth_groups = df.groupby("depth")

        # Create a list of the property values for each depth
        depths = []
        depth_values = []
        for depth, depth_group in depth_groups:
            logger.debug(f"{depth = }")
            depths.append(depth)
            depth_values.append(depth_group[prop].values)

        # Create the violin plot
        ax.violinplot(dataset=depth_values, positions=depths)

        # Set the title
        title = f"{dataset}, {metric}, {prop} vs depth"
        ax.set_title(title)

        # Set the x-axis and y-axis labels
        ax.set_xlabel("depth")
        ax.set_ylabel(prop)

        if prop.startswith("ratio"):
            # Set the y-axis limits to (-0.05, 1.05)
            ax.set_ylim(0, 1.05)

        # Tighten the layout
        fig.tight_layout()

        # Save the figure
        output_file_name = f"{prop}.png"
        output_file = output_dir.joinpath(output_file_name)
        fig.savefig(output_file, dpi=300)
