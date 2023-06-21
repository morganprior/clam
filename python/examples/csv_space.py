"""A `Space` for CSV data."""

import pathlib
import typing

import datatable
import numpy
from abd_clam import dataset
from abd_clam import metric
from abd_clam import space
from abd_clam.utils import constants
from abd_clam.utils import helpers
from scipy.special import erf

logger = helpers.make_logger(__name__)


class CsvDataset(dataset.Dataset):
    """A dataset backed by a CSV file."""

    def __init__(  # noqa PLR0913
        self,
        path: pathlib.Path,
        name: str,
        labels: typing.Union[str, list[int], numpy.ndarray],
        *,
        full_data: typing.Optional[datatable.Frame] = None,
        non_feature_columns: typing.Optional[list[str]] = None,
        indices: typing.Optional[list[int]] = None,
        normalize: typing.Optional[bool] = True,
        means: typing.Optional[numpy.ndarray] = None,
        sds: typing.Optional[numpy.ndarray] = None,
    ) -> None:
        """Initialize a `CsvDataset` object."""
        self.full_data: datatable.Frame = (
            datatable.fread(str(path)) if full_data is None else full_data
        )
        column_names = set(self.full_data.names)

        self.__path = path
        self.__name = name

        non_feature_columns = non_feature_columns or []

        if isinstance(labels, str):
            assert (
                labels in column_names
            ), f'label_column "{labels}" not found in set of column names.'
            non_feature_columns.append(labels)
            self.__labels: numpy.ndarray = (
                numpy.asarray(self.full_data[:, labels]).astype(numpy.uint).squeeze()
            )
        else:
            self.__labels = numpy.asarray(labels, dtype=numpy.uint)

        self.__non_feature_columns: list[str] = non_feature_columns or []
        for nfc in self.__non_feature_columns:
            assert (
                nfc in column_names
            ), f'non_feature_column "{nfc}" not found in set of column names.'

        self.__feature_columns = list(column_names - set(self.__non_feature_columns))
        self.__indices = (
            numpy.asarray(list(range(self.full_data.nrows)))
            if indices is None
            else numpy.asarray(indices)
        )
        self.__features: datatable.Frame = self.full_data[:, self.__feature_columns]
        self.__normalize = normalize

        means = self.__features.mean() if means is None else means
        sds = (
            (self.__features.sd() * numpy.sqrt(2) + constants.EPSILON)
            if sds is None
            else sds
        )

        fill_kwargs = {
            "nan": constants.EPSILON,
            "posinf": constants.EPSILON,
            "neginf": constants.EPSILON,
        }
        self.__means = numpy.nan_to_num(means, **fill_kwargs)
        self.__sds = numpy.nan_to_num(sds, **fill_kwargs)

        self.__shape = self.__indices.shape[0], len(self.__feature_columns)

        logger.info(f"Created CsvDataset {name} with shape {self.__shape}.")

    @property
    def name(self) -> str:
        """Return the name of the dataset."""
        return self.__name

    @property
    def path(self) -> pathlib.Path:
        """Return the path to the dataset."""
        return self.__path

    @property
    def data(self) -> datatable.Frame:
        """Return the data."""
        return self.__features

    @property
    def indices(self) -> numpy.ndarray:
        """Return the indices of the dataset."""
        return self.__indices

    @property
    def labels(self) -> numpy.ndarray:
        """Return the labels for the dataset."""
        return self.__labels

    def __eq__(self, other: "CsvDataset") -> bool:  # type: ignore[override]
        """Check if two `CsvDataset` objects are identical."""
        return self.__name == other.__name

    @property
    def max_instance_size(self) -> int:
        """Return the maximum size of an instance in bytes."""
        return 8 * self.data.shape[1]

    @property
    def approx_memory_size(self) -> int:
        """Return the approximate memory size of the dataset in bytes."""
        return self.cardinality * self.max_instance_size

    def __getitem__(
        self,
        item: typing.Union[int, typing.Iterable[int]],
    ) -> numpy.ndarray:
        """Get an instance or instances from the dataset."""
        indices = self.__indices[item]
        rows = numpy.nan_to_num(
            numpy.asarray(self.data[indices, :]),
            nan=self.__means,
            posinf=self.__means,
            neginf=self.__means,
        )

        if self.__normalize:
            rows = (1 + erf((rows - self.__means) / self.__sds)) / 2

        return rows[0] if isinstance(item, int) else rows

    def subset(
        self,
        indices: list[int],
        subset_name: str,
        labels: typing.Optional[list[int]] = None,
    ) -> "CsvDataset":
        """Return a subset of this dataset."""
        return CsvDataset(
            self.__path,
            subset_name,
            full_data=self.full_data,
            labels=self.__labels if labels is None else labels,
            non_feature_columns=self.__non_feature_columns,
            indices=indices,
            normalize=self.__normalize,
            means=self.__means,
            sds=self.__sds,
        )


class CsvSpace(space.Space):
    """A `Space` for CSV data."""

    def __init__(
        self,
        data: CsvDataset,
        distance_metric: metric.Metric,
        use_cache: bool,
    ) -> None:
        """Initialize a `CsvSpace` object."""
        super().__init__(use_cache)
        self.__data = data
        self.__distance_metric = distance_metric

    @property
    def data(self) -> CsvDataset:
        """Return the data."""
        return self.__data

    @property
    def distance_metric(self) -> metric.Metric:
        """Return the distance metric."""
        return self.__distance_metric

    def are_instances_equal(self, left: int, right: int) -> bool:
        """Return whether the instances at `left` and `right` are identical."""
        return self.distance_one_to_one(left, right) == 0.0

    def subspace(
        self,
        indices: list[int],
        subset_data_name: str,
    ) -> "CsvSpace":
        """Return a subspace of this space."""
        return CsvSpace(
            self.data.subset(indices, subset_data_name),
            self.distance_metric,
            self.uses_cache,
        )

    def distance_one_to_one(self, left: int, right: int) -> float:
        """Compute the distance between `left` and `right`."""
        return super().distance_one_to_one(left, right)

    def distance_one_to_many(
        self,
        left: int,
        right: list[int],
    ) -> numpy.ndarray:
        """Compute the distances between `left` and each instance in `right`."""
        return super().distance_one_to_many(left, right)

    def distance_many_to_many(
        self,
        left: list[int],
        right: list[int],
    ) -> numpy.ndarray:
        """Compute the distances between instances in `left` and `right`."""
        return super().distance_many_to_many(left, right)

    def distance_pairwise(self, indices: list[int]) -> numpy.ndarray:
        """Compute the distances between all pairs of instances in `indices`."""
        return super().distance_pairwise(indices)
