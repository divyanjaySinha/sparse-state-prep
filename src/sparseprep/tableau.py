"""The tableau: s basis states of n bits, split into subspace / non-subspace registers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def subspace_qubits(s: int) -> int:
    """l = ceil(log2 s), computed exactly with integers (no float error)."""
    if s < 1:
        raise ValueError("s must be >= 1")
    return (s - 1).bit_length()


def batch_size(n: int, l: int) -> int:
    """m = largest power of two with m <= n - l (paper, Section 2)."""
    free = n - l
    if free < 1:
        raise ValueError("need n - l >= 1 qubits outside the subspace register")
    return 1 << (free.bit_length() - 1)


def random_support(s: int, n: int, seed: int | None = None) -> np.ndarray:
    """s distinct random n-bit strings as an (s, n) bool array.

    Only the support matters for circuit cost; amplitudes are irrelevant.
    """
    if s > 2**n:
        raise ValueError("cannot pick s distinct strings from 2^n")
    rng = np.random.default_rng(seed)
    seen: dict[bytes, np.ndarray] = {}
    while len(seen) < s:
        cand = rng.integers(0, 2, size=(s - len(seen), n)).astype(bool)
        for row in cand:
            seen.setdefault(row.tobytes(), row)
    return np.array(list(seen.values()), dtype=bool)


@dataclass
class Tableau:
    bits: np.ndarray  # shape (s, n), dtype bool; row i is |C_i>

    def __post_init__(self) -> None:
        self.bits = np.asarray(self.bits, dtype=bool)
        if self.bits.ndim != 2:
            raise ValueError("bits must be 2-D")
        if len({r.tobytes() for r in self.bits}) != self.s:
            raise ValueError("rows must be distinct basis states")

    @property
    def s(self) -> int:
        return self.bits.shape[0]

    @property
    def n(self) -> int:
        return self.bits.shape[1]

    @property
    def l(self) -> int:
        return subspace_qubits(self.s)

    @property
    def m(self) -> int:
        return batch_size(self.n, self.l)

    @property
    def subspace(self) -> np.ndarray:
        return self.bits[:, : self.l]

    @property
    def rest(self) -> np.ndarray:
        return self.bits[:, self.l :]

    def in_subspace(self) -> np.ndarray:
        """Boolean mask: True where the non-subspace register is all zero."""
        return ~self.rest.any(axis=1)

    def row_str(self, i: int) -> str:
        r = self.bits[i]
        return "".join("1" if b else "0" for b in r[: self.l]) + "|" + \
               "".join("1" if b else "0" for b in r[self.l :])