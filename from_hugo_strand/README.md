<h1 align="center">
Spectral convolution
</h1>

> **NOTE:** The equations below may look quite intimidating, but don't worry,
> you don't have to understand them to follow along.

In this example, we will implement the imaginary time convolution for the Dyson
equation. The details can be found in [this
paper](https://doi.org/10.1063/5.0003145), but essentially we will be
implementing the recursion given by equations (10) to (12):

<div align="center">

![](./scary-convolution.png)

</div>

We will take @HugoStrand's implementation that uses
[Numba](https://numba.pydata.org/) and try to reimplement it in
[halide-haskell](https://github.com/twesterhout/halide-haskell).

Our goal is to compute a 4-dimensional tensor `mat`. It can be thought of as a
matrix of matrices. The computation proceeds as follows. We have an explicit
equation for the first column of it. The second column is then computed as a
function of the first. All the remaining columns are defined recursively in
terms of the previous ones.

Here's how one computes the first column in Python:

```python
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
```

and here's how it looks in Haskell:

```haskell
[k, n, i, j] <- mapM mkVar ["k", "n", "i", "j"]
mkColumn0 <- compile $ \s a' -> do
  a <- constantExterior 0 a'
  define "column0" (k, n, i, j) $
    bool (k `eq` 0) (s * a ! (k, i, j)) 0
      + a ! (k - 1, i, j) / (2 * cast @Double k - 1)
      - a ! (k + 1, i, j) / (2 * cast @Double k + 3)
```

There are a few differences though:

 - Variable names start with lowercase lettters (Haskell doesn't allow variable
   names to start with capitals);
 - Halide supports specifying boundary conditions such that we don't have to
   split the loop into three parts ourselves. We use `constantExterior`
   function to define `a` such that `a ! (-1, i, j)` is `0`.
 - Halide doesn't play well with recursion, so we split our implementation into
   three separate kernels: the first and second columns and the recursion. This
   is less elegant, but it turns out to perform quite well in practice
   nonetheless.
 - We have to insert explicit casts to `Double`, because halide-haskell follows
   the static typing nature of Haskell and forbids implicit casting.

Let's define the second column:

```python
# -- column 1
B[0, 1] = -s * B[1, 0] / 3.
for k in range(1, N - 1):
    B[k, 1] = B[k - 1, 0] / (2.*k - 1.) \
        - s * B[k, 0] - B[k + 1, 0] / (2.*k + 3.)
B[N - 1, 1] = B[N - 2, 0] / (2.*(N - 1) - 1.) - B[N - 1, 0]
```

and the Haskell code:

```haskell
mkColumn1 <- compile $ \s b' -> do
  b <- constantExterior 0 b'
  define "column1" (k, n, i, j) $
    bool (k `eq` 0) (-s * b ! (k + 1, 0, i, j) / 3) $
      b ! (k - 1, 0, i, j) / (2 * cast @Double k - 1)
        - s * b ! (k, 0, i, j)
        - b ! (k + 1, 0, i, j) / (2 * cast @Double k + 3)
```

The two code blocks are again quite similar.

Let's have a look at the recursion itself:

```python
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
```

The transpose at the end is done because the recursion is numerically unstable
for the upper half of the matrix `B`.

And here's the Haskell implementation:

```haskell
nextColumn :: Expr Int32 -> Func 'ParamTy 4 Double -> IO (Func 'FuncTy 4 Double)
nextColumn n b' = do
  [k, m, i, j] <- mapM mkVar ["k", "n", "i", "j"]
  b <- constantExterior 0 b'
  let upper =
        cast @Double (2 * (n + k) `mod` 2 - 1)
          * (2 * cast @Double k + 1)
          / (2 * cast @Double n + 3)
          * b ! (n + 1, k, i, j)
  let lower =
        -(2 * cast @Double n + 1) / (2 * cast @Double k + 3) * b ! (k + 1, n, i, j)
          + (2 * cast @Double n + 1) / (2 * cast @Double k - 1) * b ! (k - 1, n, i, j)
          + b ! (k, n - 1, i, j)
  define "next" (k, m, i, j) $ bool (k `lt` n) upper lower
```

It's a bit longer so we've split it into a separate function. We again rely on
the boundary conditions feature of Halide. Also note that we've decided to
merge the two loops into one and do explicit branching between the lower and
upper parts of the matrix. This better suits the Halide paradigm.

The cool this is that we can use scheduling directives to optimize the
implementation. We replace the last line by:

```haskell
f <- define "next" (k, m, i, j) $ bool (k `lt` n) upper lower
ki <- mkVar "ki"
void $ split TailShiftInwards k (k, ki) 8 f >>= vectorize ki
pure f
```

and now the look over `k` will be vectorized.

## Comparison

Let us now check that our Haskell implementation actually returns the same result as the original implementation by Hugo Strand:

```python
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
```

Our Haskell codes writes the output to `data_out_new.h5` and we assert that it
matches the result of `spectral_convolution_matrix_partial`.

We can now also compare performance. On my crappy Thinkpad L490, the results are:

```sh
$ nix develop
$ DEBUG_TIMESTATS_ENABLE=1 cabal run
kernel: 0.043s  count: 100
```

So our Haskell implementation takes about `0.43` ms.

```sh
$ python compare.py
spectral_convolution_matrix_partial:  0.00813699289021315
```

The implementation based on Numba takes about `8` ms to complete :smiley: i.e.
we got about 18 times speedup without complicating the code too much.
