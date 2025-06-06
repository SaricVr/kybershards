from numpy.typing import ArrayLike
from scipy.linalg import svd as full_svd
from sklearn.utils.extmath import randomized_svd

from kybershards.types import Seed
from kybershards.utils import check_random_state


def svd(X: ArrayLike, random_state: Seed = None) -> int:
    random_state = check_random_state(random_state)
    return 5
