from kybershards.dsci.types import LegacySeed
from kybershards.dsci.decomposition import SVDAlgorithm
from sklearn.base import ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import validate_data


class CA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int, *, copy: bool=True, algorithm=SVDAlgorithm.TRUNCATED, random_state: LegacySeed=None):
        self.n_components = n_components
        self.copy = copy
        self.algorithm = algorithm
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None) -> "CA":  # pyright: ignore [reportRedeclaration]
        X: NDArray = validate_data(self, X, ensure_min_features=2, ensure_non_negative=True)  # pyright: ignore [reportAssignmentType, reportArgumentType]
        P = X / X.sum()
        r = X.sum(axis=1)
        c = X.sum(axis=0)

        #S = sparse.diags(self.r ** -0.5) @ (X - self.r[:, np.newaxis] * self.c) @ sparse.diags(self.c ** -0.5)
        return self
