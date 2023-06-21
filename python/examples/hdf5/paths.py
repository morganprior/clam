"""Paths for the HDF5 example."""

import os
import pathlib

DATA_ROOT = pathlib.Path(
    os.environ.get(
        "DATA_ROOT",
        pathlib.Path(__file__).parent.parent.parent.parent.joinpath("data"),
    ),
).resolve()

REPORTS_DIR = DATA_ROOT.joinpath(
    "search_small",
    "reports",
)

if __name__ == "__main__":
    assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"
    assert REPORTS_DIR.exists(), f"Path not found: {REPORTS_DIR}"
