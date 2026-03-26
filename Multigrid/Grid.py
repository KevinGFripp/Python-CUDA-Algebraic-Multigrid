from numpy import ndarray
from scipy.sparse import csr_matrix
from dataclasses import dataclass,field
from numpy import float32

@dataclass(slots=True)
class Grid:
    Matrix: csr_matrix = field(default_factory=lambda: csr_matrix((0, 0),dtype=float32))
    M: csr_matrix = field(default_factory=lambda: csr_matrix((0, 0), dtype=float32))
    x: ndarray = field(default_factory=lambda: ndarray(0, dtype=float32))
    b: ndarray = field(default_factory=lambda: ndarray(0, dtype=float32))
    R: csr_matrix = field(default_factory=lambda: csr_matrix((0, 0),dtype=float32))
    I: csr_matrix = field(default_factory=lambda: csr_matrix((0, 0),dtype=float32))
    temp_array: ndarray = field(default_factory=lambda: ndarray(0, dtype=float32))
    Nx: int = 1
    Ny: int = 1