"""Utility functions."""

from numpy.random import Generator, RandomState, default_rng

from kybershards.dsci.typing import Seed


def check_random_state(seed: Seed) -> Generator:
    """Convert seed into a [`Generator`][numpy.random.Generator] instance.

    This function handles different types of random state inputs and standardizes
    them to a [`Generator`][numpy.random.Generator] for consistent random number generation.

    Args:
        seed: Source of randomness

    Returns:
        Generator: A [`Generator`][numpy.random.Generator] instance that can be used for random number generation.

    Examples:
        >>> from kybershards.dsci.utils import check_random_state
        >>>
        >>> check_random_state(42)  # Creates Generator with specific seed
        >>> check_random_state(None)  # Creates Generator with random seed
        >>> gen = np.random.default_rng(12345)  # Pass existing Generator
        >>> check_random_state(gen) is gen  # Returns the same object
        >>> rs = np.random.RandomState(12345)  # Convert legacy RandomState
        >>> check_random_state(rs)  # Returns a Generator based on the RandomState
    """
    if isinstance(seed, Generator):
        return seed
    if isinstance(seed, RandomState):
        return default_rng(seed._bit_generator)  # noqa: SLF001
    return default_rng(seed)
