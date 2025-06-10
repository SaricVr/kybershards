from enum import Enum

from numpy.typing import NDArray, ArrayLike
from scipy.linalg import svd as full_svd
from sklearn.utils.extmath import randomized_svd, svd_flip
from sklearn.utils.validation import check_random_state

from kybershards.dsci.types import LegacySeed


class SVDAlgorithm(Enum):
    FULL = "Full"
    TRUNCATED = "Truncated"


def svd(
    X: ArrayLike,
    n_components: int,
    *,
    algorithm: SVDAlgorithm = SVDAlgorithm.TRUNCATED,
    random_state: LegacySeed = None,
) -> tuple[NDArray, NDArray, NDArray]:
    if algorithm == SVDAlgorithm.FULL:
        U, S, VT = full_svd(X, full_matrices=False)
    else:
        random_state = check_random_state(random_state)
        U, S, VT = randomized_svd(X, n_components=n_components, random_state=random_state, flip_sign=False)

    U, VT = svd_flip(U, VT, u_based_decision=False)

    return U, S, VT