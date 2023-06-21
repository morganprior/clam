"""Example of searching in an HDF5 dataset."""

import logging
import math
import os
import pathlib
import time
import typing

import h5py
import numpy
from abd_clam import cluster_criteria
from abd_clam import dataset
from abd_clam import metric
from abd_clam import space
from abd_clam.search import cakes
from abd_clam.utils import helpers

logger = logging.getLogger(__name__)


class HDF5Dataset(dataset.Dataset):
    """A dataset backed by an HDF5 file."""

    def __init__(
        self,
        data: h5py.Dataset,
        name: str,
        indices: typing.Optional[list[int]] = None,
    ) -> None:
        """Initialize an `HDF5Dataset`."""
        super().__init__()
        self.__data = data
        self.__name = name
        self.__indices = numpy.asarray(indices or list(range(data.shape[0])))

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return self.__name

    @property
    def data(self) -> h5py.Dataset:
        """Return the data."""
        return self.__data

    def __eq__(self, other: "HDF5Dataset") -> bool:  # type: ignore[override]
        """Check if two datasets are identical."""
        return self.name == other.name

    @property
    def cardinality(self) -> int:
        """Return the number of instances in the dataset."""
        return len(self.__indices)

    @property
    def max_instance_size(self) -> int:
        """Return the maximum size of an instance in bytes."""
        num_bytes = self.data.dtype.itemsize
        num_features = self.data.shape[1]
        return num_features * num_bytes

    @property
    def approx_memory_size(self) -> int:
        """Return the approximate memory size of the dataset in bytes."""
        return self.cardinality * self.max_instance_size

    def __getitem__(
        self,
        item: typing.Union[int, typing.Iterable[int]],
    ) -> numpy.ndarray:
        """Get an instance or instances from the dataset."""
        if isinstance(item, int):
            item = int(self.__indices[item])
            return self.data[item]

        item_arr = numpy.asarray(item)
        indices = numpy.asarray(numpy.argsort(item_arr))
        sorted_instances = self.data[item_arr[indices]]
        instances = numpy.zeros_like(sorted_instances)
        instances[indices, :] = sorted_instances

        return instances

    def subset(
        self,
        indices: list[int],
        subset_name: str,
    ) -> "HDF5Dataset":
        """Create a subset of this dataset."""
        return HDF5Dataset(self.data, subset_name, indices)  # type: ignore[abstract]


class HDF5Space(space.Space):
    """A space for searching in an HDF5 dataset."""

    def __init__(self, data: HDF5Dataset, distance_metric: metric.Metric) -> None:
        """Initialize an `HDF5Space`."""
        super().__init__(True)
        self.__data = data
        self.__distance_metric = distance_metric

    @property
    def data(self) -> dataset.Dataset:
        """Return the dataset used by this space."""
        return self.__data

    @property
    def distance_metric(self) -> metric.Metric:
        """Return the distance metric used by this space."""
        return self.__distance_metric

    def are_instances_equal(self, left: int, right: int) -> bool:
        """Check if two instances are identical."""
        return self.distance_one_to_one(left, right) == 0.0

    def subspace(
        self,
        indices: list[int],
        subset_data_name: str,
    ) -> "HDF5Space":
        """Create a subspace of this space."""
        return HDF5Space(
            HDF5Dataset(  # type: ignore[abstract]
                self.data,
                subset_data_name,
                indices,
            ),
            self.distance_metric,
        )

    def distance_one_to_one(self, left: int, right: int) -> float:
        """Compute the distance between `left` and `right`."""
        return super().distance_one_to_one(left, right)

    def distance_one_to_many(
        self,
        left: int,
        right: list[int],
    ) -> numpy.ndarray:
        """Compute the distances between `left` and instances in `right`."""
        return super().distance_one_to_many(left, right)

    def distance_many_to_many(
        self,
        left: list[int],
        right: list[int],
    ) -> numpy.ndarray:
        """Compute the distances between instances in `left` and `right."""
        return super().distance_many_to_many(left, right)

    def distance_pairwise(self, indices: list[int]) -> numpy.ndarray:
        """Compute the pairwise distances between instances in `indices`."""
        return super().distance_pairwise(indices)


def bench_sift() -> (
    tuple[metric.ScipyMetric, HDF5Dataset, HDF5Dataset, HDF5Dataset, HDF5Dataset]
):
    """Benchmark CAKES on the sift dataset."""
    data_path = SEARCH_DATA_DIR.joinpath("as_hdf5").joinpath("sift.hdf5")

    with h5py.File(data_path, "r") as reader:
        distance_metric = metric.ScipyMetric(reader.attrs["distance"])
        train_data = HDF5Dataset(  # type: ignore[abstract]
            reader["train"],
            "sift_train",
        )
        test_data = HDF5Dataset(reader["test"], "sift_test")  # type: ignore[abstract]
        neighbors_data = HDF5Dataset(  # type: ignore[abstract]
            reader["neighbors"],
            "sift_neighbors",
        )
        distances_data = HDF5Dataset(  # type: ignore[abstract]
            reader["distances"],
            "sift_distances",
        )

        train_space = HDF5Space(train_data, distance_metric)

        start = time.perf_counter()
        searcher = cakes.CAKES(train_space).build(
            max_depth=None,
            additional_criteria=[
                cluster_criteria.MinPoints(int(math.log2(train_data.cardinality))),
            ],
        )
        end = time.perf_counter()
        build_time = end - start

        times = []
        accuracies = []
        for i in range(test_data.cardinality):
            logger.info(f"Searching query {i} ...")

            start = time.perf_counter()
            results = searcher.knn_search(test_data[i], k=100)
            end = time.perf_counter()
            times.append(end - start)

            true_hits = set(neighbors_data[i])
            matches = true_hits.intersection(results)
            accuracies.append(len(matches) / 100)

    mean_search_time = sum(times) / len(times)
    mean_search_accuracy = sum(accuracies) / len(accuracies)

    logger.info(
        f"Building CAKES took {build_time:.2e} seconds with "
        f"{train_data.cardinality} instances.",
    )
    logger.info(f"Mean search time was {mean_search_time:.2e} seconds.")
    logger.info(f"Mean search accuracy was {mean_search_accuracy:.3f}")

    return distance_metric, train_data, test_data, neighbors_data, distances_data


def convert_to_npy() -> None:
    """Convert the hdf5 files to npy files."""
    hdf5_root = SEARCH_DATA_DIR.joinpath("as_hdf5")
    npy_root = SEARCH_DATA_DIR.joinpath("as_npy")

    for hdf5_path in hdf5_root.iterdir():
        name = hdf5_path.name.split(".")[0]

        with h5py.File(hdf5_path, "r") as reader:
            for subset in ["train", "test", "neighbors", "distances"]:
                data = numpy.asarray(reader[subset])
                logger.info(f"{name}, {subset} {data.shape}, {data.dtype}")
                numpy.save(
                    npy_root.joinpath(f"{name}_{subset}.npy"),
                    data,
                    allow_pickle=False,
                    fix_imports=False,
                )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    DATA_ROOT = pathlib.Path(
        os.environ.get(
            "DATA_ROOT",
            pathlib.Path(__file__).parent.parent.parent.parent.joinpath("data"),
        ),
    ).resolve()
    assert DATA_ROOT.exists(), f"DATA_ROOT not found: {DATA_ROOT}"

    SEARCH_DATA_DIR = DATA_ROOT.joinpath("search_data")
    assert SEARCH_DATA_DIR.exists()

    logger = helpers.make_logger(__name__)

    convert_to_npy()
