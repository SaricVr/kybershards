"""Type variables."""

from numpy._typing import _ArrayLikeInt_co
from numpy.random import Generator, RandomState, SeedSequence

Seed = Generator | RandomState | SeedSequence | _ArrayLikeInt_co | None
"""`_ArrayLikeInt_co` is any array sequence of `int` that can be converted to a numpy array.
    This is defined in the numpy package
"""

LegacySeed = int | RandomState | None
