import numpy as np
import pytest

from sparseprep.tableau import Tableau, batch_size, random_support, subspace_qubits


@pytest.mark.parametrize("s,l", [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (1000, 10), (1024, 10), (1025, 11)])
def test_subspace_qubits(s, l):
    assert subspace_qubits(s) == l


@pytest.mark.parametrize("n,l,m", [(7, 3, 4), (6, 3, 2), (8, 3, 4), (4, 3, 1), (13, 3, 8)])
def test_batch_size(n, l, m):
    assert batch_size(n, l) == m


def test_random_support_is_distinct_and_reproducible():
    a = random_support(50, 12, seed=1)
    b = random_support(50, 12, seed=1)
    assert a.shape == (50, 12)
    assert np.array_equal(a, b)
    assert len({r.tobytes() for r in a}) == 50


def test_paper_example_fig3():
    # The 7 states from Figures 3-5: 3-bit subspace | 4-bit rest.
    strings = ["0111011", "1110111", "1001010", "0110110", "1011101", "1001111", "1111010"]
    t = Tableau(np.array([[c == "1" for c in x] for x in strings]))
    assert (t.s, t.n, t.l, t.m) == (7, 7, 3, 4)
    assert t.row_str(0) == "011|1011"
    assert not t.in_subspace().any()


def test_in_subspace_mask():
    t = Tableau(np.array([[1, 0, 0, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=bool))
    assert t.in_subspace().tolist() == [True, False, True]


def test_duplicate_rows_rejected():
    with pytest.raises(ValueError):
        Tableau(np.array([[1, 0], [1, 0]], dtype=bool))