"""Methods to performs regularisation to spectra"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear


def tikhonov_smoothing(
    vector: NDArray, lambda_param: float = 0.1, differential: int = 2
):
    """Method to apply tikhonov smoothing

    Solving the least square system of equations

        min x ||Ax-b||2 + lambda ||Lx||2

    Parameters
    ----------
    vector : NDArray
        A vector of values for which Tikhonov regression is to be applied.
    lambda_param : float
        The lambda parameter represents how aggressively smoothing is applied,
        by default 0.1
    differential : int
        The order of the differential applied, by default 2
    """

    # set up initial forward operator (mat_a) and measured data (mat_b)
    phi_length: int = len(vector)
    mat_a: NDArray = np.eye(phi_length)
    mat_b: NDArray = vector.copy()

    # calculate the l_matrix for a given differential
    l_matrix: NDArray = np.diff(np.eye(phi_length), axis=0, n=differential)

    # calculate the augmented matrices
    mat_a_aug: NDArray = np.vstack([mat_a, np.sqrt(lambda_param) * l_matrix])
    mat_b_aug: NDArray = np.concatenate([mat_b, np.zeros(l_matrix.shape[0])])

    # apply bound to ensure positive values: x >= 0
    bounds: tuple[NDArray, NDArray] = (
        np.zeros(phi_length),
        np.full(phi_length, np.inf),
    )

    # solve augmented system of terms
    res = lsq_linear(mat_a_aug, mat_b_aug, bounds=bounds, method="trf")

    # return the solution vector
    return res.x
