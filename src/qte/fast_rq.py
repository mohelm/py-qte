import os
import numpy as np
import scipy.optimize
from typing import Optional
from numpy.typing import NDArray

try:
    from qte import rq_fortran
    HAS_FORTRAN = True
except ImportError:
    HAS_FORTRAN = False

def fast_quantreg(X: NDArray[np.float64], y: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """
    Computes a single Quantile Regression model.
    Uses Koenker's bare-metal Fortran solver if compiled (0.003s per model).
    Falls back to SciPy Dual HiGHS LP solver if Fortran isn't compiled (0.02s per model).
    """
    n, p = X.shape
    
    if HAS_FORTRAN:
        # Roger Koenker's Frisch-Newton Algorithm (Directly from R `quantreg`)
        a = np.asfortranarray(X.T)
        y_input = -y
        rhs = (1 - q) * X.sum(axis=0)
        
        d = np.ones(n)
        u = np.ones(n)
        beta = 0.99995
        eps = 1e-6
        
        wn = np.zeros((n, 9), order='F')
        wn[:, 0] = 1 - q
        
        wp = np.zeros((p, p + 3), order='F')
        nit = np.zeros(3, dtype=np.int32)
        info = 0
        
        rq_fortran.rqfnb(a, y_input, rhs, d, u, beta, eps, wn, wp, nit, info)
        return -wp[:, 0]
        
    else:
        # Scipy HiGHS Dual LP Solver (Pure Python fallback)
        c = -y
        A_eq = X.T
        b_eq = np.zeros(p)
        bounds = [(q - 1.0, q)] * n
        
        res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        return -res.eqlin.marginals
