import numpy as np
from numpy.linalg import norm
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import diags
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from kybershards.dsci.decomposition import SVDAlgorithm
from kybershards.dsci.decomposition._svd import svd
from kybershards.dsci.typing import LegacySeed

from enum import Enum


class Coordinates(Enum):
    ROWS = "Rows"
    COLUMNS = "Columns"


class CA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Correspondence Analysis (CA).

    Correspondence Analysis is a multivariate statistical technique that provides a graphical
    representation of contingency tables. It finds a low-dimensional representation of the association
    between rows and columns of the table, similar to how PCA operates but for categorical data.

    The algorithm decomposes the standardized residuals from independence into principal dimensions
    that explain the maximum inertia (chi-square distance).

    Args:
        n_components: Number of dimensions to keep in the results. When set to a number `n` between `0` and `1`,
            the `n_components` is set to the minimum number that explain >= `n` of inertia.
        compute_contributions: Whether to compute the contributions of rows and columns
            to the principal dimensions.
        algorithm: SVD algorithm to use for the decomposition.
        random_state: Controls the randomness of the SVD solver.

    References:
        [1] Benzécri, J.-P. (1973). L'Analyse des Données. Volume II. L'Analyse des
               Correspondances. Paris, France: Dunod.

        [2] Greenacre, M. J. (1984). Theory and Applications of Correspondence Analysis.
               London: Academic Press.
    """

    def __init__(
        self,
        n_components: float,
        *,
        compute_contributions: bool = True,
        algorithm: SVDAlgorithm = SVDAlgorithm.TRUNCATED,
        random_state: LegacySeed = None,
    ):
        self.n_components = n_components
        self.compute_contributions = compute_contributions
        self.algorithm = algorithm
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "CA":  # pyright: ignore [reportRedeclaration]
        """Fit the CA model to the data.

        Args:
            X: Input contingency table or frequency matrix with non-negative values.
            y: Ignored. Kept for API compatibility with scikit-learn.

        Returns:
            The fitted estimator
        """
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
            S,
            int(self.n_components) if self.n_components >= 1 else X.shape[1],
            algorithm=self.algorithm,
            random_state=self.random_state,
        )

        # Inertia
        self.inertia_ = s**2
        # This needs to consider all the features in input regardless of n_components
        self.total_inertia_ = norm(S, "fro") ** 2
        self.explained_inertia_ = self.inertia_ / self.total_inertia_

        # Retrieve the number of components based on the exlained inertia
        if 0 < self.n_components < 1:
            cumulative_inertia = self.explained_inertia_.cumsum()
            n_components = (cumulative_inertia >= self.n_components).argmax() + 1
            # Truncate the results to keep only those components
            self.row_components_ = self.row_components_[:, :n_components]
            self.column_components_ = self.column_components_[:n_components, :]
            self.inertia_ = self.inertia_[:n_components]
            self.explained_inertia_ = self.explained_inertia_[:n_components]
            s = s[:n_components]

        if self.compute_contributions:
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

        return self

    def transform(self, X: ArrayLike, coordinates: Coordinates = Coordinates.ROWS) -> NDArray:  # pyright: ignore [reportRedeclaration]
        """Transform the data to the principal component space.

        The transformation projects the input data onto the principal components
        learned during the fit step.

        Args:
            X: Input contingency table or frequency matrix with non-negative values.
            coordinates: Which coordinates to compute in the princopal dimensions.

        Returns:
            Coordinates of rows or columns in the principal dimensions, based on `coordinates` argument
        """
        check_is_fitted(self)
        X: NDArray = validate_data(self, X, ensure_min_features=2, ensure_non_negative=True)  # pyright: ignore [reportAssignmentType, reportArgumentType]
        return self._transform(X, coordinates=coordinates)

    def _transform(self, X: NDArray, coordinates: Coordinates = Coordinates.ROWS) -> NDArray:
        if coordinates == Coordinates.ROWS:
            return self._row_coordinates(X)
        return self._column_coordinates(X.T)

    def fit_transform(
        self,
        X: ArrayLike,  # pyright: ignore [reportRedeclaration]
        y: ArrayLike | None = None,
        coordinates: Coordinates = Coordinates.ROWS,
        **fit_params,
    ) -> NDArray:
        """Fit the CA model and transform the data in a single step.

        Args:
            X: Input contingency table or frequency matrix with non-negative values.
            y: Ignored. Kept for API compatibility with scikit-learn.
            coordinates: Which coordinates to compute in the princopal dimensions.

        Returns:
            Returns:
            Coordinates of rows or columns in the principal dimensions, based on `coordinates` argument
        """
        X: NDArray = validate_data(self, X, ensure_min_features=2, ensure_non_negative=True)  # pyright: ignore [reportAssignmentType, reportArgumentType]
        return self._fit(X)._transform(X, coordinates=coordinates)

    def _row_coordinates(self, X: NDArray) -> NDArray:
        return (X / X.sum(axis=1)[:, np.newaxis]) @ self.diag_c_ @ self.column_components_.T

    def _column_coordinates(self, X: NDArray) -> NDArray:
        return (X / X.sum(axis=1)[:, np.newaxis]) @ self.diag_r_ @ self.row_components_

    def row_cosine_similarity(self, X: ArrayLike, F: ArrayLike) -> NDArray:  # pyright: ignore [reportRedeclaration]
        """Compute the cosine similarity between rows in original space and principal dimensions.

        The cosine similarity represents the quality of representation of the rows
        in each principal dimension. It measures the proportion of the chi-square distance
        explained by each dimension.

        Args:
            X: Input contingency table or frequency matrix with non-negative values.
            F: Row coordinates in the principal dimensions, which should be the output of
                `transform(X, coordinates=Coordinates.ROWS)`.

        Returns:
            Cosine similarity values for each row across all dimensions.
        """
        check_is_fitted(self)
        X: NDArray = validate_data(self, X, ensure_min_features=2, ensure_non_negative=True)

        # Chi-square distances
        row_profiles = X / X.sum(axis=1, keepdims=True)
        column_margins = X.sum(axis=0) / X.sum()
        chi_sq_distances = ((row_profiles - column_margins) ** 2 / column_margins).sum(axis=1)[:, np.newaxis]

        cos2 = np.divide(F**2, chi_sq_distances, out=np.zeros_like(F), where=chi_sq_distances > 0)

        return cos2

    def column_cosine_similarity(self, X: ArrayLike, G: ArrayLike) -> NDArray:  # pyright: ignore [reportRedeclaration]
        """Compute the cosine similarity (cos²) between columns in original space and principal dimensions.

        The cosine similarity represents the quality of representation of the columns
        in each principal dimension. It measures the proportion of the chi-square distance
        explained by each dimension.

        Args:
            X: Input contingency table or frequency matrix with non-negative values.
            G: Column coordinates in the principal dimensions, which should be the output of
                `transform(X, coordinates=Coordinates.COLUMNS)`.

        Returns:
            Cosine similarity values for each column across all dimensions.
        """
        check_is_fitted(self)
        X: NDArray = validate_data(self, X, ensure_min_features=2, ensure_non_negative=True)

        # Chi-square distances
        col_profiles = X / X.sum(axis=0, keepdims=True)
        row_margins = X.sum(axis=1) / X.sum()
        chi_sq_distances = ((col_profiles.T - row_margins) ** 2 / row_margins).sum(axis=1)[:, np.newaxis]

        cos2 = np.divide(G**2, chi_sq_distances, out=np.zeros_like(G), where=chi_sq_distances > 0)

        return cos2

    @property
    def row_components(self) -> NDArray:
        """Gets the row components (left singular vectors) from the CA decomposition.

        Returns:
            The row components matrix.
        """
        check_is_fitted(self)
        return self.row_components_

    @property
    def column_components(self) -> NDArray:
        """Gets the column components (right singular vectors) from the CA decomposition.

        Returns:
            The column components matrix.
        """
        check_is_fitted(self)
        return self.column_components_

    @property
    def inertia(self) -> NDArray:
        """Gets the inertia (eigenvalues) of the principal dimensions.

        The inertia represents the amount of variance explained by each
        principal dimension, equivalent to the squared singular values.

        Returns:
            Inertia values for each principal dimension.
        """
        check_is_fitted(self)
        return self.inertia_

    @property
    def total_inertia(self) -> float:
        """Gets the total inertia of the contingency table.

        The total inertia is the chi-square statistic of the table divided
        by the sum of all elements. It represents the total variance in
        the data that can potentially be explained.

        Returns:
            Total inertia of the contingency table.
        """
        check_is_fitted(self)
        return self.total_inertia_

    @property
    def explained_inertia(self) -> NDArray:
        """Gets the proportion of inertia explained by each principal dimension.

        Each value represents the proportion of the total inertia (total variance)
        that is explained by the corresponding principal dimension.

        Returns:
            Proportion of inertia explained by each principal dimension.
        """
        check_is_fitted(self)
        return self.explained_inertia_

    @property
    def row_contributions(self) -> NDArray:
        """Gets the contributions of rows to each principal dimension.

        The contributions represent how much each row contributes to the
        definition of each principal dimension. They sum to 1 across all
        rows for a given dimension.

        Returns:
            Contribution values of rows to dimensions.
        """
        check_is_fitted(
            self,
            "row_contributions_",
            msg="The estimator has to be fitted with compute_contributions=True "
            "in order to retrieve the row contributions",
        )
        return self.row_contributions_

    @property
    def column_contributions(self) -> NDArray:
        """Gets the contributions of columns to each principal dimension.

        The contributions represent how much each column contributes to the
        definition of each principal dimension. They sum to 1 across all
        columns for a given dimension.

        Returns:
            Contribution values of columns to dimensions.
        """
        check_is_fitted(
            self,
            "column_contributions_",
            msg="The estimator has to be fitted with compute_contributions=True "
            "in order to retrieve the column contributions",
        )
        return self.column_contributions_
