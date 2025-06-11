import numpy as np
from numpy.linalg import norm
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import diags
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils.validation import validate_data, check_is_fitted

from kybershards.dsci.decomposition import SVDAlgorithm
from kybershards.dsci.decomposition._svd import svd
from kybershards.dsci.types import LegacySeed


class CA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    def __init__(
        self, n_components: int, *, copy: bool = True, algorithm=SVDAlgorithm.TRUNCATED, random_state: LegacySeed = None
    ):
        self.n_components = n_components
        self.copy = copy
        self.algorithm = algorithm
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None) -> "CA":  # pyright: ignore [reportRedeclaration]
        X: NDArray = validate_data(self, X, ensure_min_features=2, ensure_non_negative=True)  # pyright: ignore [reportAssignmentType, reportArgumentType]
        return self._fit(X)

    def _fit(self, X: NDArray) -> "CA":
        # Correspondence matrix
        P = X / X.sum()

        self.row_masses_ = P.sum(axis=1)
        self.col_masses_ = P.sum(axis=0)

        # Diagonal matrices of the marginal distributions (we can use sparse)
        eps = np.finfo(float).eps
        self.diag_r_ = diags(np.maximum(self.row_masses_, eps) ** -0.5)
        self.diag_c_ = diags(np.maximum(self.col_masses_, eps) ** -0.5)

        # Residuals from expected frequencies under independence
        D = P - self.row_masses_[:, np.newaxis] * self.col_masses_

        # Standardize residuals
        S = self.diag_r_ @ D @ self.diag_c_

        # Left singular vector, singular values, right singular vector transposed
        self.row_components_, s, self.column_components_ = svd(
            S, self.n_components, algorithm=self.algorithm, random_state=self.random_state
        )

        # Inertia
        self.inertia_ = s**2
        # This needs to consider all the features in input regardless of n_components
        self.total_inertia_ = norm(S, "fro") ** 2
        self.explained_inertia_ = self.inertia_ / self.total_inertia_

        Ds = diags(s)

        # Principal coordinates for rows
        row_principal = self.diag_r_ @ self.row_components_ @ Ds
        self.row_contributions_ = np.divide(
            self.row_masses_[:, np.newaxis] * row_principal**2,
            self.inertia_[np.newaxis, :],
            out=np.zeros_like(row_principal),
            where=s > 0,
        )

        # Principal coordinates for columns
        column_principal = self.diag_c_ @ self.column_components_.T @ Ds
        self.column_contributions_ = np.divide(
            self.col_masses_[:, np.newaxis] * column_principal**2,
            self.inertia_[np.newaxis, :],
            out=np.zeros_like(column_principal),
            where=s > 0,
        )

        self.row_cosine_similarity_ = self._row_cosine_similarity(X)
        self.column_cosine_similarity_ = self._column_cosine_similarity(X)

        return self

    def transform(self, X: ArrayLike, row_coordinates: bool = True, column_coordinates: bool = False) -> NDArray:  # pyright: ignore [reportRedeclaration]
        check_is_fitted(self)
        X: NDArray = validate_data(self, X, ensure_min_features=2, ensure_non_negative=True)  # pyright: ignore [reportAssignmentType, reportArgumentType]
        return self._transform(X, row_coordinates=row_coordinates, column_coordinates=column_coordinates)

    def _transform(self, X: NDArray, row_coordinates: bool = True, column_coordinates: bool = False) -> NDArray:
        if not row_coordinates ^ column_coordinates:
            raise ValueError(
                "Transform method should compute either row_coordinates or column_coordinates, not both or none. "
                f"The value passed are {row_coordinates, column_coordinates}"
            )

        if row_coordinates:
            return self._row_coordinates(X)
        return self._column_coordinates(X.T)

    def fit_transform(
        self,
        X: ArrayLike,  # pyright: ignore [reportRedeclaration]
        y=None,
        row_coordinates: bool = True,
        column_coordinates: bool = False,
        **fit_params,
    ) -> NDArray:
        X: NDArray = validate_data(self, X, ensure_min_features=2, ensure_non_negative=True)  # pyright: ignore [reportAssignmentType, reportArgumentType]
        return self._fit(X)._transform(X, row_coordinates=row_coordinates, column_coordinates=column_coordinates)

    def _row_coordinates(self, X: NDArray) -> NDArray:
        return (X / X.sum(axis=1)[:, np.newaxis]) @ self.diag_c_ @ self.column_components_.T

    def _column_coordinates(self, X: NDArray) -> NDArray:
        return (X / X.sum(axis=1)[:, np.newaxis]) @ self.diag_r_ @ self.row_components_
    
    def row_cosine_similarity(self, X: ArrayLike) -> NDArray:  # pyright: ignore [reportRedeclaration]
        check_is_fitted(self)
        X: NDArray = validate_data(self, X, ensure_min_features=2, ensure_non_negative=True)
        
        return self._row_cosine_similarity(X)
    
    def _row_cosine_similarity(self, X: NDArray) -> NDArray:
        F = self._row_coordinates(X)
        
        # Chi-square distances
        row_profiles = X / X.sum(axis=1, keepdims=True)
        column_margins = X.sum(axis=0) / X.sum()
        chi_sq_distances = ((row_profiles - column_margins)**2 / column_margins).sum(axis=1)[:, np.newaxis]
        
        cos2 = np.divide(
            F**2,
            chi_sq_distances, 
            out=np.zeros_like(F), 
            where=chi_sq_distances > 0
        )
        
        return cos2
    
    def column_cosine_similarity(self, X: ArrayLike) -> NDArray:  # pyright: ignore [reportRedeclaration]
        check_is_fitted(self)
        X: NDArray = validate_data(self, X, ensure_min_features=2, ensure_non_negative=True)
        
        return self._column_cosine_similarity(X)
    
    def _column_cosine_similarity(self, X: NDArray) -> NDArray:
        G = self._column_coordinates(X.T)
        
        # Chi-square distances
        col_profiles = X / X.sum(axis=0, keepdims=True)
        row_margins = X.sum(axis=1) / X.sum()
        chi_sq_distances = ((col_profiles.T - row_margins)**2 / row_margins).sum(axis=1)[:, np.newaxis]
        
        cos2 = np.divide(
            G**2, 
            chi_sq_distances, 
            out=np.zeros_like(G), 
            where=chi_sq_distances > 0
        )
        
        return cos2
    
    @property
    def inertia(self):
        check_is_fitted(self)
        return self.inertia_
    
    @property
    def total_inertia(self):
        check_is_fitted(self)
        return self.total_inertia_
    
    @property
    def explained_inertia(self):
        check_is_fitted(self)
        return self.explained_inertia_
    
    @property
    def row_contributions(self):
        check_is_fitted(self)
        return self.row_contributions_
    
    @property
    def column_contributions(self):
        check_is_fitted(self)
        return self.column_contributions_
