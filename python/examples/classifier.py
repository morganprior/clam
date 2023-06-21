"""Example for running a CLAM classifier."""

import logging
import pathlib
import time

import datatable
import numpy
from abd_clam import metric
from abd_clam.classification import classifier
from abd_clam.utils import synthetic_data
from sklearn.metrics import accuracy_score
from utils import paths

from . import csv_space

logger = logging.getLogger(__name__)


def make_bullseye(path: pathlib.Path, n: int, force: bool = False) -> None:
    """Make a bullseye (multiple concentric rings) dataset."""
    if not force and path.exists():
        return

    data, labels = synthetic_data.bullseye(n=n, num_rings=3, noise=0.10)
    x = data[:, 0].astype(numpy.float32)
    y = data[:, 1].astype(numpy.float32)
    labels = numpy.asarray(labels, dtype=numpy.int8)

    full = datatable.Frame({"x": x, "y": y, "label": labels})
    full.to_csv(str(path))


# noinspection DuplicatedCode
def main() -> None:
    """Run the example."""
    bullseye_train = csv_space.CsvDataset(
        BULLSEYE_TRAIN_PATH,
        "bullseye_train",
        labels=LABEL_COLUMN,
    )
    bullseye_spaces = [
        csv_space.CsvSpace(bullseye_train, metric.ScipyMetric("euclidean"), False),
        csv_space.CsvSpace(bullseye_train, metric.ScipyMetric("cityblock"), False),
    ]

    start = time.perf_counter()
    bullseye_classifier = classifier.Classifier(
        bullseye_train.labels,
        bullseye_spaces,
    ).build()
    end = time.perf_counter()
    build_time = end - start

    bullseye_test = csv_space.CsvDataset(
        BULLSEYE_TEST_PATH,
        "bullseye_test",
        labels=LABEL_COLUMN,
    )

    start = time.perf_counter()
    predicted_labels, _ = bullseye_classifier.predict(bullseye_test)
    end = time.perf_counter()
    prediction_time = end - start

    score = accuracy_score(bullseye_test.labels, predicted_labels)

    logger.info("Building the classifier for:")
    logger.info(f"\t{bullseye_train.cardinality} instances and")
    logger.info(f"\t{bullseye_classifier.unique_labels} unique labels")
    logger.info(f"\ttook {build_time:.2e} seconds.")

    logger.info("Predicting from the classifier for:")
    logger.info(f"\t{bullseye_test.cardinality} instances took")
    logger.info(f"\ttook {prediction_time:.2e} seconds.")

    logger.info(f"The accuracy score was {score:.3f}")

    # Desktop   non-cached, cached
    # build,    152,        154
    # search,   105,        106
    # accuracy, 0.999,      1.000

    # M1Pro     non-cached, cached
    # build,    95.7,       96.1
    # search,   48.4,       48.7
    # accuracy, 0.999,      0.999


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    SYNTHETIC_DATA_DIR = paths.DATA_ROOT.joinpath("synthetic_data")
    SYNTHETIC_DATA_DIR.mkdir(exist_ok=True)

    BULLSEYE_TRAIN_PATH = SYNTHETIC_DATA_DIR.joinpath("bullseye_train.csv")
    BULLSEYE_TEST_PATH = SYNTHETIC_DATA_DIR.joinpath("bullseye_test.csv")
    FEATURE_COLUMNS = ["x", "y"]
    LABEL_COLUMN = "label"

    make_bullseye(BULLSEYE_TRAIN_PATH, n=1000, force=True)
    make_bullseye(BULLSEYE_TEST_PATH, n=200, force=True)
    main()
