"""Setup variants for the generality harness (tests/check_general.py).

Every case returns a dict with the inputs `differometor.run_build_step` needs plus
the initial values of the optimised parameters. Cases deliberately cover paths the
scored fixture never exercises: other topologies, multiple signal/noise detectors,
homodyne (qhd) phases, laser amplitude/frequency modulation, carrier-changing sweeps,
different frequency counts/ranges and higher-finesse (worse conditioned) cavities.
"""
import copy
import json
from pathlib import Path

import numpy as np

import differometor as df
from differometor.setups import aligo, constrain_inter_grid_cell_spaces, uifo, voyager

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]


def _values(data, pairs):
    values = []
    for pair in pairs:
        name, prop = pair[0] if isinstance(pair[0], list) else pair
        section = "nodes" if name in data["nodes"] else "edges"
        values.append(float(data[section][name]["properties"][prop]))
    return np.asarray(values, dtype=np.float64)


def _freqs(lo, hi, n):
    return np.logspace(np.log10(lo), np.log10(hi), n)


def _uifo_case(data, freqs):
    setup = df.Setup.from_data(data)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    return dict(setup=setup, changing_pairs=[("f", "frequency")], changing_values=freqs,
                pairs=pairs, initial=_values(data, pairs))


def _generic_case(setup, freqs, changing=("f", "frequency"), properties=PROPERTIES):
    pairs = [list(p) for p in setup.parameters if p[1] in properties]
    data = setup.to_data()
    return dict(setup=setup, changing_pairs=[changing], changing_values=freqs,
                pairs=pairs, initial=_values(data, pairs))


def _jitter(data, seed, finesse_boost=False):
    """Perturb every optimisable property of a UIFO fixture (topology unchanged)."""
    rng = np.random.default_rng(seed)
    data = copy.deepcopy(data)
    length_scale = {}
    for name, node in data["nodes"].items():
        props = node.get("properties", {})
        if "reflectivity" in props:
            r = props["reflectivity"]
            if finesse_boost and 1e-3 < r < 1 - 1e-12:
                r = 1 - (1 - r) * 0.2          # 5x lower transmission -> higher finesse
            else:
                r = float(np.clip(r ** rng.uniform(0.9, 1.1), 1e-12, 1 - 1e-12))
            props["reflectivity"] = r
        if "tuning" in props:
            props["tuning"] += rng.uniform(-3, 3)
        if "mass" in props:
            props["mass"] *= rng.uniform(0.7, 1.4)
        if "power" in props:
            props["power"] *= rng.uniform(0.5, 1.5)
        if "db" in props:
            props["db"] = max(0.0, props["db"] + rng.uniform(-2, 2))
        if "angle" in props:
            props["angle"] += rng.uniform(-10, 10)
    for name, edge in data["edges"].items():
        props = edge["properties"]
        # scale lengths per *value* so tied grid lengths stay equal
        key = props["length"]
        length_scale.setdefault(key, rng.uniform(0.8, 1.2))
        props["length"] = key * length_scale[key]
    return data


def _load(name):
    return json.loads((DATA / name).read_text())


def scored():
    data = _load("uifo_800_3000.json")
    ref = _load("uifo_800_3000_test_data.json")
    return _uifo_case(data, np.asarray(ref["frequencies"]))


CASES = {
    # the exact benchmark workload
    "uifo3_scored": scored,
    # same topology, perturbed parameters, different frequency grid
    "uifo3_jitter": lambda: _uifo_case(_jitter(_load("uifo_800_3000.json"), 1), _freqs(500, 4000, 37)),
    # same topology, 5x lower mirror transmissions -> worse conditioning
    "uifo3_finesse": lambda: _uifo_case(_jitter(_load("uifo_800_3000.json"), 2, finesse_boost=True), _freqs(800, 3000, 23)),
    # other trained UIFO: 2 signal detectors, 2 noise detectors, qhd phase, 7 squeezers
    "uifo3_20_5000": lambda: _uifo_case(_load("uifo_20_5000.json"), _freqs(20, 5000, 29)),
    # random 2x2 UIFOs: different topology + modulation modes
    "uifo2_rand_space": lambda: _generic_case(uifo(2, random=True, random_seed=3)[0], _freqs(100, 3000, 17)),
    "uifo2_rand_am": lambda: _generic_case(uifo(2, random=True, random_seed=5, mode="amplitude_modulation")[0], _freqs(100, 3000, 17)),
    "uifo2_rand_fm": lambda: _generic_case(uifo(2, random=True, random_seed=7, mode="frequency_modulation")[0], _freqs(100, 3000, 17)),
    # classic detectors
    "aligo_space": lambda: _generic_case(aligo("space_modulation")[0], np.asarray(_load("aligo_test_data.json")["frequencies"])),
    "aligo_am": lambda: _generic_case(aligo("amplitude_modulation")[0], _freqs(5, 5000, 41)),
    "aligo_fm": lambda: _generic_case(aligo("frequency_modulation")[0], _freqs(5, 5000, 41)),
    "voyager_space": lambda: _generic_case(voyager("space_modulation")[0], _freqs(20, 5000, 50)),
    "voyager_fm": lambda: _generic_case(voyager("frequency_modulation")[0], _freqs(20, 5000, 50)),
    # carrier-changing sweep (tuning), several power detectors, no signal / noise detector
    "cavity_tuning": lambda: _cavity(),
}


def _cavity():
    S = df.Setup()
    S.add("laser", "l0", power=1)
    S.add("mirror", "m0", reflectivity=0.99, loss=0)
    S.add("mirror", "m1", reflectivity=0.991, loss=0)
    S.space("l0", "m0", length=1)
    S.space("m0", "m1", length=1)
    S.add("detector", "refl", target="m0", port="left", direction="out")
    S.add("detector", "circ", target="m1", port="left", direction="in")
    S.add("detector", "trns", target="m1", port="right", direction="out")
    case = _generic_case(S, np.linspace(-180, 180, 61), changing=("m0", "tuning"),
                         properties=["reflectivity", "power", "length"])
    return case
