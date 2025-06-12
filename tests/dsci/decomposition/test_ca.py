import pytest
import numpy as np
from numpy.typing import NDArray

from pytest import FixtureRequest

from kybershards.dsci.decomposition import CA, Coordinates, SVDAlgorithm


@pytest.fixture(scope="module")
def contingency_table() -> NDArray:
    rng = np.random.default_rng(42)
    return rng.integers(100, size=(10, 15))


@pytest.fixture(scope="module", params=[SVDAlgorithm.TRUNCATED, SVDAlgorithm.FULL])
def ca(contingency_table: NDArray, request: FixtureRequest) -> NDArray:
    ca = CA(n_components=0.6, algorithm=request.param, compute_contributions=True, random_state=42)
    return ca.fit(contingency_table)


@pytest.fixture(scope="module", params=[SVDAlgorithm.TRUNCATED, SVDAlgorithm.FULL])
def ca_fit_transform(request: FixtureRequest) -> CA:
    return CA(n_components=0.6, algorithm=request.param, compute_contributions=False, random_state=42)


@pytest.fixture(scope="module")
def row_projection(ca: CA, contingency_table: NDArray) -> NDArray:
    return ca.transform(contingency_table)


@pytest.fixture(scope="module")
def column_projection(ca: CA, contingency_table: NDArray) -> NDArray:
    return ca.transform(contingency_table, coordinates=Coordinates.COLUMNS)


@pytest.fixture
def row_components() -> NDArray:
    return np.array([[-0.0320043 ,  0.6379847 ,  0.25726034],
       [ 0.13961609, -0.21031199,  0.17134229],
       [-0.46741268, -0.19054613,  0.70743537],
       [ 0.05930979,  0.24480569, -0.09597509],
       [-0.27856009, -0.37249016, -0.23097146],
       [ 0.41159199, -0.0920982 , -0.09811652],
       [-0.29549759,  0.20655365, -0.45536429],
       [-0.20995506, -0.34123633, -0.29163507],
       [ 0.06308252,  0.2950921 , -0.09456396],
       [ 0.61246042, -0.24308381,  0.17452293]])


@pytest.fixture
def column_components() -> NDArray:
    return np.array([[-0.11252485, -0.04199533, -0.21186618, -0.07513414, -0.13914001,
         0.50446667, -0.34942356, -0.31055776,  0.20963131,  0.28265276,
         0.16296285, -0.23332365, -0.22516313,  0.43239172,  0.01252598],
       [-0.2871438 ,  0.02797665,  0.3438644 , -0.17360035,  0.01739586,
         0.14846015, -0.53167878,  0.15615646, -0.05124847, -0.33796713,
        -0.15944733,  0.42747108,  0.0443638 ,  0.13193127,  0.3075688 ],
       [-0.19141508, -0.23423785, -0.44089256,  0.01917319,  0.00190386,
         0.02382434,  0.1171434 ,  0.14454078, -0.09870864,  0.23652603,
        -0.22900158,  0.54090529,  0.25964796,  0.30223595, -0.33027466]])


@pytest.fixture
def row_coordinates() -> NDArray:
    return np.array(
        [
            [-0.02692787, 0.5239472, 0.1745797],
            [0.11517826, -0.16934901, 0.11400577],
            [-0.42425079, -0.16881299, 0.51788836],
            [0.04934742, 0.19881223, -0.06440562],
            [-0.23768328, -0.31022595, -0.15895174],
            [0.34739926, -0.07587465, -0.06679306],
            [-0.24956898, 0.17027588, -0.31018648],
            [-0.18824244, -0.29862767, -0.21089106],
            [0.05232597, 0.23891831, -0.06326465],
            [0.56894679, -0.22041099, 0.13075953],
        ]
    )


@pytest.fixture
def column_coordinates() -> NDArray:
    return np.array(
        [
            [-0.11277614, -0.28090005, -0.15472915],
            [-0.04121671, 0.02680104, -0.18542],
            [-0.23682284, 0.375174, -0.39748609],
            [-0.07496859, -0.16907375, 0.01542992],
            [-0.12985244, 0.01584629, 0.00143305],
            [0.51016058, 0.14654395, 0.01943222],
            [-0.409149, -0.60766218, 0.11063036],
            [-0.33509652, 0.16446408, 0.12578972],
            [0.22713588, -0.05419935, -0.0862605],
            [0.29090262, -0.3395099, 0.19633628],
            [0.1627476, -0.15542713, -0.18445553],
            [-0.28678376, 0.51284499, 0.53622152],
            [-0.2542356, 0.04889353, 0.23645647],
            [0.46464116, 0.13837946, 0.26194713],
            [0.01311536, 0.31433629, -0.27891452],
        ]
    )


@pytest.fixture
def inertia() -> NDArray:
    return np.array([0.07366123, 0.07017883, 0.04791742])


@pytest.fixture
def total_inertia() -> float:
    return 0.2839566051549266


@pytest.fixture
def explained_inertia() -> NDArray:
    return np.array([0.25941018, 0.24714634, 0.16874909])


@pytest.fixture
def row_contributions() -> NDArray:
    return np.array(
        [
            [0.00102428, 0.40702448, 0.06618288],
            [0.01949265, 0.04423113, 0.02935818],
            [0.21847461, 0.03630783, 0.5004648],
            [0.00351765, 0.05992983, 0.00921122],
            [0.07759572, 0.13874892, 0.05334782],
            [0.16940796, 0.00848208, 0.00962685],
            [0.08731882, 0.04266441, 0.20735664],
            [0.04408113, 0.11644223, 0.08505101],
            [0.0039794, 0.08707935, 0.00894234],
            [0.37510777, 0.05908974, 0.03045825],
        ]
    )


@pytest.fixture
def column_contributions() -> NDArray:
    return np.array(
        [
            [1.26618417e-02, 8.24515635e-02, 3.66397312e-02],
            [1.76360780e-03, 7.82693142e-04, 5.48673701e-02],
            [4.48872784e-02, 1.18242725e-01, 1.94386253e-01],
            [5.64513910e-03, 3.01370824e-02, 3.67611155e-04],
            [1.93599425e-02, 3.02615950e-04, 3.62468185e-06],
            [2.54486622e-01, 2.20404154e-02, 5.67599200e-04],
            [1.22096827e-01, 2.82682323e-01, 1.37225755e-02],
            [9.64461213e-02, 2.43848389e-02, 2.08920373e-02],
            [4.39452872e-02, 2.62640591e-03, 9.74339614e-03],
            [7.98925811e-02, 1.14221781e-01, 5.59445626e-02],
            [2.65568889e-02, 2.54234521e-02, 5.24417249e-02],
            [5.44399245e-02, 1.82731528e-01, 2.92578530e-01],
            [5.06984364e-02, 1.96814643e-03, 6.74170627e-02],
            [1.86962602e-01, 1.74058602e-02, 9.13465702e-02],
            [1.56900125e-04, 9.45985692e-02, 1.09081352e-01],
        ]
    )


@pytest.fixture
def row_cosine_similarity() -> NDArray:
    return np.array(
        [
            [0.00195514, 0.74020018, 0.08217914],
            [0.07723341, 0.16696656, 0.07566898],
            [0.36104303, 0.05716436, 0.53800455],
            [0.01088098, 0.17661405, 0.01853474],
            [0.19785837, 0.3370648, 0.08848868],
            [0.60550765, 0.02888386, 0.02238332],
            [0.21416528, 0.09969517, 0.33083662],
            [0.13540001, 0.34075614, 0.16994169],
            [0.0205947, 0.42935816, 0.03010531],
            [0.67984013, 0.10203049, 0.03590957],
        ]
    )


@pytest.fixture
def column_cosine_similarity() -> NDArray:
    return np.array(
        [
            [5.69280020e-02, 3.53179194e-01, 1.07160757e-01],
            [1.51868066e-02, 6.42130256e-03, 3.07349865e-01],
            [1.31417074e-01, 3.29814770e-01, 3.70210300e-01],
            [5.85223161e-02, 2.97656396e-01, 2.47907674e-03],
            [1.84443214e-01, 2.74674022e-03, 2.24637855e-05],
            [8.30628655e-01, 6.85375983e-02, 1.20514182e-03],
            [2.72892616e-01, 6.01940014e-01, 1.99515880e-02],
            [4.52255790e-01, 1.08939760e-01, 6.37286315e-02],
            [2.10684472e-01, 1.19963526e-02, 3.03868032e-02],
            [3.00048132e-01, 4.08696161e-01, 1.36677425e-01],
            [1.18155968e-01, 1.07765597e-01, 1.51778364e-01],
            [1.23924406e-01, 3.96296258e-01, 4.33247652e-01],
            [1.66430035e-01, 6.15547683e-03, 1.43966484e-01],
            [6.42095983e-01, 5.69518525e-02, 2.04075978e-01],
            [5.77680197e-04, 3.31830203e-01, 2.61257720e-01],
        ]
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
    ca: CA, contingency_table: NDArray, row_projection: NDArray, row_cosine_similarity: NDArray
) -> None:
    np.testing.assert_array_almost_equal(
        ca.row_cosine_similarity(contingency_table, row_projection), row_cosine_similarity
    )


def test_ca_column_cosine_similarity(
    ca: CA, contingency_table: NDArray, column_projection: NDArray, column_cosine_similarity: NDArray
) -> None:
    np.testing.assert_array_almost_equal(
        ca.column_cosine_similarity(contingency_table, column_projection), column_cosine_similarity
    )


def test_ca_fit_transform(ca_fit_transform: CA, contingency_table: NDArray, row_coordinates: NDArray) -> None:
    np.testing.assert_array_almost_equal(ca_fit_transform.fit_transform(contingency_table), row_coordinates)
