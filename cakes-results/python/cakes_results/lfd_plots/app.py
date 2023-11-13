"""CLI command to create the lfd-plots for the Cakes tres."""

import logging
import pathlib

import tqdm
import typer

from . import create_plots as _create_plots

# Initialize the logger
logger = logging.getLogger("lfd-plots")
logger.setLevel("INFO")

app = typer.Typer()


@app.command()
def create_plots(
    input_dir: pathlib.Path = typer.Option(
        ...,
        "--input-dir",
        "-i",
        help="The directory containing the csvs of trees.",
        exists=True,
        readable=True,
        file_okay=False,
        resolve_path=True,
    ),
    output_dir: pathlib.Path = typer.Option(
        ...,
        "--output-dir",
        "-o",
        help="The directory to save the plots.",
        exists=True,
        writable=True,
        file_okay=False,
        resolve_path=True,
    ),
) -> None:
    """Create the plots for the scaling results of the Cakes search."""
    logger.info(f"input_dir = {input_dir}")
    logger.info(f"output_dir = {output_dir}")

    tree_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    # tree_dirs = [p for p in tree_dirs if p.name.startswith("mnist")]

    for tree_dir in tqdm.tqdm(tree_dirs, desc="Processing trees"):
        _create_plots(tree_dir, output_dir)
