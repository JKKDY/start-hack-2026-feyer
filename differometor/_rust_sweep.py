"""Packed coefficient sweep using the optional native library."""
import ctypes
from .components import FUNCTIONS
import numpy as np

class Sweep:
    def __init__(self, instruction, varying, width, library):
        groups, order = instruction
        n = sum(len(g[1]) for g in groups if g is not None)
        destination = np.arange(n)
        if order is not None:
            if not np.array_equal(np.sort(order), destination):
                raise ValueError("Invalid coefficient permutation")
            destination = np.argsort(order)
        plans, writes = [], []
        offset = 0
        for kind, group in enumerate(groups):
            if group is None:
                continue
            if kind >= len(FUNCTIONS):
                raise ValueError("Unknown optical component")
            inputs, rows, cols = group
            if inputs.shape[1] != 7 or np.any((inputs < 0) | (inputs >= width)):
                raise ValueError("Invalid optical input indices")
            if np.any((rows < 0) | (rows >= len(inputs))) or np.any((cols < 0) | (cols >= 4)):
                raise ValueError("Invalid optical output indices")
            masks = np.isin(inputs, varying) @ (1 << np.arange(7))
            pos = np.argsort(rows, kind="stable")
            ends = np.cumsum(np.bincount(rows, minlength=len(inputs))) + offset
            starts = np.r_[offset, ends[:-1]]
            plans.append(np.column_stack((np.full(len(inputs), kind), masks, starts, ends, inputs)))
            writes.append(np.column_stack((cols[pos], destination[offset + pos])))
            offset += len(rows)
        self.width = width
        self.plan, self.writes = (np.ascontiguousarray(np.concatenate(a), dtype=np.int32) for a in (plans, writes))
        self.kernel = ctypes.CDLL(str(library)).run_coefficients
        self.kernel.argtypes = [ctypes.c_bool] + [ctypes.c_size_t]*4 + [ctypes.c_void_p]*5
        self.kernel.restype = None

    def __call__(self, parameters, seed):
        x = np.ascontiguousarray(parameters, dtype=np.complex128)
        if x.ndim != 2 or x.shape[1] != self.width:
            raise ValueError("Invalid parameter shape")
        n = len(self.writes)
        if seed is not None:
            seed = np.ascontiguousarray(seed, dtype=np.complex128)
            if seed.shape != (len(x), n):
                raise ValueError("Invalid coefficient cotangent")
        out = np.empty(x.shape if seed is not None else (len(x), n), dtype=x.dtype)
        if len(x):
            self.kernel(seed is not None, *x.shape, len(self.plan), n, x.ctypes.data,
                        self.plan.ctypes.data, self.writes.ctypes.data,
                        None if seed is None else seed.ctypes.data, out.ctypes.data)
        return out
