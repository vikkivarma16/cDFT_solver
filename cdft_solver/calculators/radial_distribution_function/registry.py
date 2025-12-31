# cdft_solver/closures/registry.py

from .builtin import py_closure, hnc_closure, hybrid_closure

CLOSURE_REGISTRY = {
    "PY": py_closure,
    "HNC": hnc_closure,
    "HYBRID": hybrid_closure,
}

