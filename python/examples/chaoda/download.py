"""Download and save the datasets."""

import logging
import pathlib

from . import anomaly_data

logger = logging.getLogger(__name__)


def download_and_save(data_dir: pathlib.Path) -> None:
    """Download and save the datasets."""
    for name, url in sorted(anomaly_data.DATASET_URLS.items()):
        data = anomaly_data.AnomalyData(data_dir, name, url).download().preprocess()
        save_path = data.save()

        logger.info(f"Saved {data.name} to {save_path}")


def load(data_dir: pathlib.Path) -> None:
    """Load the datasets."""
    for name in anomaly_data.DATASET_URLS:
        data = anomaly_data.AnomalyData.load(data_dir, name)

        logger.info(
            f"loaded {data.name} data with features of shape {data.features.shape}.",
        )
