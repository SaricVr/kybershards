import pytest
from numpy.random import Generator, RandomState, SeedSequence, default_rng

from kybershards.dsci.typing import Seed
from kybershards.dsci.utils import check_random_state


@pytest.mark.parametrize(
    "seed", [42, None, default_rng(12345), RandomState(12345), [1, 2, 3, 4, 5], SeedSequence([1, 2, 3, 4, 5])]
)
def test_check_random_state(seed: Seed) -> None:
    result = check_random_state(seed)
    if isinstance(seed, Generator):
        assert result is seed, "Should return the same Generator object"
    else:
        assert isinstance(result, Generator)
