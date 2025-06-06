from numpy.random import Generator, default_rng

from kybershards.types import Seed


def check_random_state(seed: Seed) -> Generator:
    if isinstance(seed, Generator):
        return seed
    return default_rng(seed)
