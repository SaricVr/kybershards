from numpy._typing import _ArrayLikeInt_co
from numpy.random import Generator, RandomState, SeedSequence

Seed = Generator | RandomState | SeedSequence | _ArrayLikeInt_co | None
LegacySeed = int | RandomState | None
