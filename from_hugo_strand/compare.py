import h5py
import numpy as np
import timeit
from recursion import spectral_convolution_matrix_partial

def main():
    with h5py.File("data_in.h5", "r") as f:
        G = np.array(f["G"])

    # Call once to ensure that the kernel is compiled
    mat = spectral_convolution_matrix_partial(G, +1.)

    time = timeit.timeit(lambda: spectral_convolution_matrix_partial(G, +1.), number=100)
    print("spectral_convolution_matrix_partial: ", time / 100)

    with h5py.File("data_out_new.h5", "r") as f:
        mat_haskell = np.array(f["mat"])

    np.testing.assert_array_almost_equal(mat, mat_haskell)

if __name__ == '__main__':
    main()
