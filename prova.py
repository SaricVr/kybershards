import numpy as np
import pandas as pd
from light_famd import CA as CAP
from prince import CA as CAPR

from kybershards.dsci.decomposition import CA, SVDAlgorithm, Coordinates

rng = np.random.default_rng(42)
X = rng.integers(100, size=(10, 15))

ca = CA(n_components=0.6, algorithm=SVDAlgorithm.FULL, compute_contributions=True)
ca.fit(X)
print(repr(ca.total_inertia))

X = pd.DataFrame(X)

capr = CAPR(n_components=7, random_state=42)
capr.fit(X)
print(capr.row_contributions_.shape)

# cap = CAP(n_components=15)
# cap.fit(X)
# print(cap.row_contributions_.shape)
