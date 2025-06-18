import numpy as np
import pandas as pd
import pytest
import sklearn
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from kybershards.dsci.datasets import load_workers_smoking_habits
from kybershards.dsci.decomposition import CA, Coordinates, SVDAlgorithm


@pytest.fixture(scope="module")
def contingency_table() -> pd.DataFrame:
    return load_workers_smoking_habits()


@pytest.fixture(scope="module")
def ca(contingency_table: pd.DataFrame) -> CA:
    ca = CA(n_components=0.99, compute_contributions=True, random_state=42)
    return ca.fit(contingency_table.values)


@pytest.fixture(scope="module")
def ca_pipeline(contingency_table: pd.DataFrame) -> pd.DataFrame:
    with sklearn.config_context(transform_output="pandas"):
        pipeline = Pipeline(
            [
                ("identity_in", FunctionTransformer()),
                ("ca", CA(n_components=2, algorithm=SVDAlgorithm.FULL)),
                ("identity_out", FunctionTransformer()),
            ],
        )
        return pipeline.fit_transform(contingency_table)


@pytest.fixture(scope="module")
def row_projection(ca: CA, contingency_table: pd.DataFrame) -> NDArray:
    return ca.transform(contingency_table)


@pytest.fixture(scope="module")
def column_projection(ca: CA, contingency_table: pd.DataFrame) -> NDArray:
    return ca.transform(contingency_table, coordinates=Coordinates.COLUMNS)


@pytest.fixture
def row_components() -> NDArray:
    return np.array(
        [
            [0.05742524, 0.46212293],
            [-0.28923816, 0.74239515],
            [0.71554563, 0.05475038],
            [-0.57530335, -0.38957951],
            [0.2646963, -0.28376408],
        ],
    )


@pytest.fixture
def column_components() -> NDArray:
    return np.array(
        [[0.80870009, -0.17564112, -0.40696007, -0.38670126], [0.17127755, -0.68056865, -0.04167443, 0.71116353]],
    )


@pytest.fixture
def row_coordinates() -> NDArray:
    return np.array(
        [
            [0.06576838, 0.193737],
            [-0.25895842, 0.24330457],
            [0.38059489, 0.01065991],
            [-0.23295191, -0.05774391],
            [0.20108912, -0.07891123],
        ],
    )


@pytest.fixture
def column_coordinates() -> NDArray:
    return np.array(
        [[0.39330845, 0.03049207], [-0.09945592, -0.14106429], [-0.19632096, -0.00735911], [-0.29377599, 0.19776566]],
    )


@pytest.fixture
def inertia() -> NDArray:
    return np.array([0.07475911, 0.01001718])


@pytest.fixture
def total_inertia() -> float:
    return 0.08518986047784068


@pytest.fixture
def explained_inertia() -> NDArray:
    return np.array([0.87755873, 0.11758654])


@pytest.fixture
def row_contributions() -> NDArray:
    return np.array(
        [
            [0.00329766, 0.2135576],
            [0.08365871, 0.55115055],
            [0.51200555, 0.0029976],
            [0.33097395, 0.15177219],
            [0.07006413, 0.08052205],
        ],
    )


@pytest.fixture
def column_contributions() -> NDArray:
    return np.array(
        [[0.65399583, 0.029336], [0.0308498, 0.46317368], [0.1656165, 0.00173676], [0.14953787, 0.50575356]],
    )


@pytest.fixture
def row_cosine_similarity() -> NDArray:
    return np.array(
        [
            [9.22320257e-02, 8.00336390e-01],
            [5.26399913e-01, 4.64682461e-01],
            [9.99032948e-01, 7.83719694e-04],
            [9.41934118e-01, 5.78762422e-02],
            [8.65345506e-01, 1.33256997e-01],
        ],
    )


@pytest.fixture
def column_cosine_similarity() -> NDArray:
    return np.array(
        [[0.99402039, 0.00597451], [0.32672616, 0.65728966], [0.98184805, 0.00137963], [0.68439774, 0.31015425]],
    )


def test_ca_row_components(ca: CA, row_components: NDArray) -> None:
    np.testing.assert_array_almost_equal(ca.row_components, row_components)


def test_ca_column_components(ca: CA, column_components: NDArray) -> None:
    np.testing.assert_array_almost_equal(ca.column_components, column_components)


def test_ca_row_coordinates(row_projection: NDArray, row_coordinates: NDArray) -> None:
    np.testing.assert_array_almost_equal(row_projection, row_coordinates)


def test_ca_column_coordinates(column_projection: NDArray, column_coordinates: NDArray) -> None:
    np.testing.assert_array_almost_equal(column_projection, column_coordinates)


def test_ca_inertia(ca: CA, inertia: NDArray) -> None:
    np.testing.assert_array_almost_equal(ca.inertia, inertia)


def test_ca_total_inertia(ca: CA, total_inertia: NDArray) -> None:
    np.testing.assert_almost_equal(ca.total_inertia, total_inertia)


def test_ca_explained_inertia(ca: CA, explained_inertia: NDArray) -> None:
    np.testing.assert_array_almost_equal(ca.explained_inertia, explained_inertia)


def test_ca_row_contributions(ca: CA, row_contributions: NDArray) -> None:
    np.testing.assert_array_almost_equal(ca.row_contributions, row_contributions)


def test_ca_column_contributions(ca: CA, column_contributions: NDArray) -> None:
    np.testing.assert_array_almost_equal(ca.column_contributions, column_contributions)


def test_ca_row_cosine_similarity(
    ca: CA,
    contingency_table: pd.DataFrame,
    row_projection: NDArray,
    row_cosine_similarity: NDArray,
) -> None:
    np.testing.assert_array_almost_equal(
        ca.row_cosine_similarity(contingency_table, row_projection),
        row_cosine_similarity,
    )


def test_ca_column_cosine_similarity(
    ca: CA,
    contingency_table: pd.DataFrame,
    column_projection: NDArray,
    column_cosine_similarity: NDArray,
) -> None:
    np.testing.assert_array_almost_equal(
        ca.column_cosine_similarity(contingency_table, column_projection),
        column_cosine_similarity,
    )


def test_ca_fit_transform(ca_pipeline: pd.DataFrame, row_coordinates: NDArray) -> None:
    np.testing.assert_array_almost_equal(ca_pipeline, row_coordinates)


def test_ca_pandas_output(ca_pipeline: pd.DataFrame, contingency_table: pd.DataFrame) -> None:
    np.testing.assert_array_equal(ca_pipeline.index, contingency_table.index)
    np.testing.assert_array_equal(ca_pipeline.columns.values, [f"ca{i}" for i in range(len(ca_pipeline.columns))])


def test_ca_n_components_range(contingency_table: pd.DataFrame) -> None:
    ca = CA(n_components=0)
    with pytest.raises(ValueError, match="n_components=0 must be between 1 and n_features or between 0 and 1"):
        ca.fit(contingency_table)
