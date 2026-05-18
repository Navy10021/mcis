"""Compatibility shims for dependencies across versions."""

import numpy as np


def _apply_patches():
    """Apply runtime patches for known dependency incompatibilities.

    NumPy >= 2.0 removed ``np.MachAr``, but older statsmodels (< 0.15)
    still references it at import time.  This shim restores a minimal
    replacement so that statsmodels can be imported normally.

    SciPy >= 1.17 moved ``_centered`` from ``scipy.signal.signaltools``
    to ``scipy.signal._signaltools``; older statsmodels still imports
    from the old location.
    """
    if not hasattr(np, "MachAr"):
        class _MachAr:
            class _EPS:
                def __init__(self):
                    import sys
                    self.eps = sys.float_info.epsilon
            eps = _EPS().eps
        np.MachAr = lambda: _MachAr()

    try:
        from scipy.signal._signaltools import _centered
    except ImportError:
        _centered = None

    if _centered is not None:
        try:
            import scipy.signal.signaltools as _st
            if not hasattr(_st, "_centered"):
                _st._centered = _centered
        except ImportError:
            pass


_apply_patches()
