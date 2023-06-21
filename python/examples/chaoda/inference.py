"""Run CHAODA inference on the datasets."""

import json
import math
import pathlib
import time
import typing

import abd_clam
import sklearn

from . import anomaly_data

logger = abd_clam.utils.helpers.make_logger(__name__)


def run_one_dataset(
    data_dir: pathlib.Path,
    name: str,
    metrics: typing.Sequence[abd_clam.Metric],
    output_dir: pathlib.Path,
) -> None:
    """Run CHAODA inference on a single dataset."""
    raw_data = anomaly_data.AnomalyData.load(data_dir, name)
    dataset = abd_clam.dataset.TabularDataset(
        data=raw_data.normalized_features,
        name=name,
    )

    spaces = [abd_clam.space.TabularSpace(dataset, metric, False) for metric in metrics]

    min_cardinality: int = 1 + int(math.log2(dataset.cardinality))
    # if dataset.cardinality < 10_000:

    criteria = [abd_clam.cluster_criteria.MinPoints(min_cardinality)]

    start = time.perf_counter()
    chaoda = abd_clam.anomaly_detection.CHAODA(
        spaces,
        partition_criteria=criteria,
    )
    predicted_scores = chaoda.fit_predict()
    time_taken = time.perf_counter() - start

    roc_score = sklearn.metrics.roc_auc_score(raw_data.scores, predicted_scores)

    logger.info(f"Dataset {name} scored {roc_score:.3f} in {time_taken:.2e} seconds.")

    results = {
        "roc_score": f"{roc_score:.6f}",
        "time_taken": f"{time_taken:.2e} seconds",
        "predicted_scores": [f"{s:.6f}" for s in predicted_scores],
    }
    results_path = output_dir.joinpath(name)
    results_path.mkdir(exist_ok=True)

    with results_path.joinpath("results.json").open("w") as writer:
        json.dump(results, writer, indent=4)


def compile_results(output_dir: pathlib.Path) -> None:
    """Compile the results into a single file."""
    full_results: dict[str, dict[str, typing.Any]] = {}
    for name in anomaly_data.INFERENCE_SET:
        full_results[name] = {}

        with output_dir.joinpath(name).joinpath("results.json").open("r") as reader:
            results = json.load(reader)
        full_results[name]["roc_score"] = results["roc_score"]
        full_results[name]["time_taken"] = results["time_taken"]

    with output_dir.joinpath("full_results.json").open("w") as writer:
        json.dump(full_results, writer, indent=4)


def run_inference(data_dir: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Run CHAODA inference on the datasets."""
    metrics = [
        abd_clam.metric.ScipyMetric("euclidean"),
        abd_clam.metric.ScipyMetric("cityblock"),
    ]

    for name in anomaly_data.INFERENCE_SET:
        logger.info(f"Staring CHAODA inference on {name} ...")
        run_one_dataset(data_dir, name, metrics, output_dir)

    compile_results(output_dir)
