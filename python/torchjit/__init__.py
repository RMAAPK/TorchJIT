import ctypes
import os
from typing import List, Sequence

_so_path = os.path.join(os.path.dirname(__file__), "libtorchjit.so")
if not os.path.exists(_so_path):
    raise RuntimeError(f"TorchJIT backend library not found at {_so_path}")

_lib = ctypes.CDLL(_so_path)

# C-API bindings
_lib.tjit_tensor_zeros.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
_lib.tjit_tensor_zeros.restype = ctypes.c_void_p

_lib.tjit_tensor_ones.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
_lib.tjit_tensor_ones.restype = ctypes.c_void_p

_lib.tjit_tensor_load_mmap.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
_lib.tjit_tensor_load_mmap.restype = ctypes.c_void_p

_lib.tjit_tensor_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.tjit_tensor_save.restype = None

_lib.tjit_tensor_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.tjit_tensor_matmul.restype = ctypes.c_void_p

_lib.tjit_tensor_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
_lib.tjit_tensor_add.restype = ctypes.c_void_p

_lib.tjit_tensor_rmsnorm.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.tjit_tensor_rmsnorm.restype = ctypes.c_void_p

_lib.tjit_tensor_data_ptr.argtypes = [ctypes.c_void_p]
_lib.tjit_tensor_data_ptr.restype = ctypes.POINTER(ctypes.c_float)

_lib.tjit_tensor_numel.argtypes = [ctypes.c_void_p]
_lib.tjit_tensor_numel.restype = ctypes.c_int64

_lib.tjit_tensor_cid.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
_lib.tjit_tensor_cid.restype = None

_lib.tjit_tensor_free.argtypes = [ctypes.c_void_p]
_lib.tjit_tensor_free.restype = None


class Tensor:
    """TorchJIT Sovereign Tensor Object"""
    def __init__(self, handle, shape: Sequence[int]):
        self._handle = handle
        self._shape = list(shape)

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.tjit_tensor_free(self._handle)
            self._handle = None

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def numel(self) -> int:
        return _lib.tjit_tensor_numel(self._handle)

    @property
    def cid(self) -> str:
        buf = ctypes.create_string_buffer(128)
        _lib.tjit_tensor_cid(self._handle, buf, 128)
        return buf.value.decode('utf-8')

    def tolist(self) -> List[float]:
        ptr = _lib.tjit_tensor_data_ptr(self._handle)
        n = self.numel
        return [ptr[i] for i in range(n)]

    def __add__(self, other: 'Tensor') -> 'Tensor':
        out_handle = _lib.tjit_tensor_add(self._handle, other._handle)
        return Tensor(out_handle, self._shape)

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        out_handle = _lib.tjit_tensor_matmul(self._handle, other._handle)
        out_shape = [self._shape[0], other._shape[1]]
        return Tensor(out_handle, out_shape)

    def rms_norm(self, eps: float = 1e-6) -> 'Tensor':
        out_handle = _lib.tjit_tensor_rmsnorm(self._handle, ctypes.c_float(eps))
        return Tensor(out_handle, self._shape)

    def save(self, filepath: str) -> None:
        _lib.tjit_tensor_save(self._handle, filepath.encode('utf-8'))

    def __repr__(self):
        return f"torchjit.Tensor(shape={self._shape}, cid={self.cid[:18]}...)"


def zeros(shape: Sequence[int]) -> Tensor:
    c_shape = (ctypes.c_int64 * len(shape))(*shape)
    handle = _lib.tjit_tensor_zeros(c_shape, len(shape))
    return Tensor(handle, shape)


def ones(shape: Sequence[int]) -> Tensor:
    c_shape = (ctypes.c_int64 * len(shape))(*shape)
    handle = _lib.tjit_tensor_ones(c_shape, len(shape))
    return Tensor(handle, shape)


def load_mmap(filepath: str, shape: Sequence[int]) -> Tensor:
    c_shape = (ctypes.c_int64 * len(shape))(*shape)
    handle = _lib.tjit_tensor_load_mmap(filepath.encode('utf-8'), c_shape, len(shape))
    return Tensor(handle, shape)
