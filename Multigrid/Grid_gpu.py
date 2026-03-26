from cupy import ndarray as gpu_array
from cupyx.scipy.sparse import csr_matrix as csr_matrix_gpu
from dataclasses import dataclass,field
from numpy import float32

@dataclass(slots=True)
class Grid_GPU:
    Matrix: csr_matrix_gpu = field(default_factory=lambda: csr_matrix_gpu((0, 0),dtype=float32))
    M: csr_matrix_gpu = field(default_factory=lambda: csr_matrix_gpu((0, 0), dtype=float32))
    x: gpu_array = field(default_factory=lambda: gpu_array(0, dtype=float32))
    b: gpu_array = field(default_factory=lambda: gpu_array(0, dtype=float32))
    R: csr_matrix_gpu = field(default_factory=lambda: csr_matrix_gpu((0, 0),dtype=float32))
    I: csr_matrix_gpu = field(default_factory=lambda: csr_matrix_gpu((0, 0),dtype=float32))
    temp_array: gpu_array = field(default_factory=lambda: gpu_array(0, dtype=float32))
    Nx: int = 1
    Ny: int = 1

