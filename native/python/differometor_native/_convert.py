"""Conversion of Python setup/request objects into native types.

This is the single place where Python data crosses into the native planner:
- `setup_spec(setup)` snapshots a `Setup` (or its `to_data()` dict) into an
  immutable native `SetupSpec`. Later mutations of the Python setup do not
  affect plans built from the snapshot.
- `normalize_*` turn upstream's flexible argument forms into plain lists.
"""

from __future__ import annotations

import math
import numbers
import sys
from typing import Any, Iterable

import numpy as np

from differometor_native import _native

_NATIVE_LINKS = frozenset(_native.link_functions())

# Callables recognised as parameter links, by identity. JAX is never imported
# here; if the caller already imported jax.numpy, its functions are added.
_LINKS_BY_IDENTITY: dict[int, str] = {
    id(np.sqrt): "sqrt",
    id(math.sqrt): "sqrt",
    id(np.square): "square",
    id(np.exp): "exp",
    id(math.exp): "exp",
    id(np.log): "log",
    id(math.log): "log",
    id(np.sin): "sin",
    id(math.sin): "sin",
    id(np.cos): "cos",
    id(math.cos): "cos",
    id(np.negative): "negative",
    id(np.abs): "abs",
    id(abs): "abs",
}

_JAX_LINK_NAMES = {
    "sqrt": "sqrt",
    "square": "square",
    "exp": "exp",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
    "negative": "negative",
    "abs": "abs",
    "absolute": "abs",
}


def link_function_name(function: Any) -> str:
    """Map a link callable onto a native link function name.

    Raises ValueError for callables the native engine cannot evaluate and
    differentiate (lambdas, user functions). See docs/api-compatibility.md.
    """
    name = _LINKS_BY_IDENTITY.get(id(function))
    if name is not None:
        return name
    jnp = sys.modules.get("jax.numpy")
    if jnp is not None:
        for attr, native_name in _JAX_LINK_NAMES.items():
            if getattr(jnp, attr, None) is function:
                return native_name
    raise ValueError(
        f"Unsupported parameter-link callable {function!r}. The native engine supports "
        f"{sorted(_NATIVE_LINKS)} (from numpy, math or jax.numpy)."
    )


def _property_value(owner: str, key: str, value: Any) -> Any:
    if isinstance(value, tuple):
        if len(value) != 2 or not isinstance(value[0], str) or not callable(value[1]):
            raise ValueError(
                f"Linked property {owner}.{key} must be (parameter_name, callable), got {value!r}."
            )
        return (value[0], link_function_name(value[1]))
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (numbers.Real, np.floating, np.integer)):
        raise ValueError(f"Property {owner}.{key} must be a real number or a link, got {value!r}.")
    return float(value)


def _properties(owner: str, props: dict) -> list:
    return [(key, _property_value(owner, key, value)) for key, value in props.items()]


def setup_spec(setup_or_data: Any):
    """Snapshot a `Setup` or a `Setup.to_data()` dict into a native SetupSpec."""
    data = setup_or_data if isinstance(setup_or_data, dict) else setup_or_data.to_data()
    nodes = []
    for name, node in data["nodes"].items():
        nodes.append(
            (
                name,
                node["component"],
                _properties(name, node.get("properties", {})),
                node.get("target"),
                node.get("target_property"),
                node.get("port"),
                node.get("direction"),
                node.get("auxiliary"),
                node.get("detector1"),
                node.get("detector2"),
                node.get("sideband"),
            )
        )
    edges = []
    for edge_name, edge in data["edges"].items():
        source, target = edge_name.split("_", 1)
        edges.append(
            (
                source,
                target,
                _properties(edge_name, edge.get("properties", {})),
                edge.get("source_port", "right"),
                edge.get("target_port", "left"),
            )
        )
    parameters = [(str(a), str(b)) for a, b in data["parameters"]]
    return _native.core.SetupSpec.from_normalized(nodes, edges, parameters)


def normalize_pairs(pairs: Iterable | None) -> list[tuple[str, str]]:
    """(name, property) pairs as a list of string tuples."""
    if pairs is None:
        return []
    return [(str(pair[0]), str(pair[1])) for pair in pairs]


def normalize_optimization(pairs: Iterable | None) -> list[list[tuple[str, str]]]:
    """Upstream optimization entries -> list of groups.

    An entry is either one (name, property) pair, or a list of pairs tied to a
    single optimisation variable (as produced by
    `setups.constrain_inter_grid_cell_spaces`).
    """
    if pairs is None:
        return []
    groups = []
    for entry in pairs:
        if len(entry) > 0 and isinstance(entry[0], str):
            groups.append([(str(entry[0]), str(entry[1]))])
        else:
            groups.append([(str(p[0]), str(p[1])) for p in entry])
    return groups


def normalize_changing_values(values: Any, n_pairs: int) -> np.ndarray | None:
    """Sweep values as a C-contiguous float64 array of shape (n_pairs, N)."""
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1 and n_pairs == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(
            f"changing_values must have shape (len(changing_pairs), num_values); got {array.shape}."
        )
    return np.ascontiguousarray(array)
