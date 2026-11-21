"""Compile-time constant index arrays.

`run_build_step` returns a dict of arrays that callers (e.g. benchmark.py) pass to
`jax.jit` as an *argument*. Every leaf of a jit argument becomes a tracer, so
integer index arrays (which entry uses which component function, where it is
written in which matrix, ...) would be unknown at compile time. XLA then has to
emit fully general gathers/scatters and cannot specialise on the setup structure.

`StaticArray` is a pytree node with *no* children: its value travels in the
treedef's auxiliary data. jit therefore treats it as a compile-time constant (and
as part of the compilation cache key), while all float/complex data stays dynamic.
Everything here is derived from whatever setup was built, so it is setup-agnostic.
"""

import numpy as np
import jax


@jax.tree_util.register_pytree_node_class
class StaticArray:
    """A hashable, read-only NumPy array that jit sees as static structure."""

    __slots__ = ("array", "_hash")

    def __init__(self, array):
        array = np.array(array, copy=True)
        array.setflags(write=False)
        self.array = array
        self._hash = hash((array.dtype.str, array.shape, array.tobytes()))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, StaticArray)
            and self._hash == other._hash
            and self.array.dtype == other.array.dtype
            and self.array.shape == other.array.shape
            and np.array_equal(self.array, other.array)
        )

    def __repr__(self):
        return f"StaticArray(shape={self.array.shape}, dtype={self.array.dtype})"

    def tree_flatten(self):
        return (), self

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return aux_data


def _is_static_candidate(leaf):
    dtype = getattr(leaf, "dtype", None)
    return dtype is not None and (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_))


def make_static(tree):
    """Wrap every integer/bool array leaf of `tree` in a StaticArray."""
    return jax.tree_util.tree_map(
        lambda leaf: StaticArray(np.asarray(leaf)) if _is_static_candidate(leaf) else leaf, tree
    )


def unwrap_static(tree):
    """Replace StaticArray nodes by their NumPy arrays (compile-time constants under jit)."""
    return jax.tree_util.tree_map(
        lambda node: node.array if isinstance(node, StaticArray) else node,
        tree,
        is_leaf=lambda node: isinstance(node, StaticArray),
    )
