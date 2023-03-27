
""" Author: H. U.R. Strand, 2020 """

# ----------------------------------------------------------------------

import numpy as np
import numba as nb

# ----------------------------------------------------------------------
@nb.jit(nb.float64[:,:,:,:](nb.float64[:,:,:], nb.float64), nopython=True, cache=True)
def spectral_convolution_matrix_partial(A_naa, s):

    N = A_naa.shape[0]
    Na = A_naa.shape[1]

    B = np.zeros((N, N, Na, Na), dtype=A_naa.dtype)
    a_n = A_naa

    # -- column 0
    B[0, 0] = s * a_n[0] - a_n[1] / 3.
    for k in range(1, N - 1):
        B[k, 0] = a_n[k - 1] / (2.*k - 1.) - a_n[k + 1] / (2.*k + 3.)
    B[N - 1, 0] = a_n[N - 2] / (2.*(N - 1) - 1.)

    # -- column 1
    B[0, 1] = -s * B[1, 0] / 3.
    for k in range(1, N - 1):
        B[k, 1] = B[k - 1, 0] / (2.*k - 1.) \
            - s * B[k, 0] - B[k + 1, 0] / (2.*k + 3.)
    B[N - 1, 1] = B[N - 2, 0] / (2.*(N - 1) - 1.) - B[N - 1, 0]

    # -- Recurse columns
    for n in range(1, N - 1):
        for k in range(n + 1, N - 1):
            B[k, n + 1] = -(2.*n + 1.) / (2.*k + 3.) * B[k + 1, n] \
                + (2.*n + 1) / (2.*k - 1.) * B[k - 1, n] + B[k, n - 1]
        k = N - 1
        B[N - 1, n + 1] = (2.*n + 1) / (2.*k - 1.) * B[k - 1, n] + B[k, n - 1]

    # -- Transpose tril to triu        
    for k in range(N - 1):
        for n in range(N):
            B[k, n] = (-1)**(n + k) * (2.*k + 1.) / (2.*n + 1.) * B[n, k]

    return B

