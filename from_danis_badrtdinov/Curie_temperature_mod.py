import numpy as np
from numba import jit

J_r = np.zeros((3, 3, 11, 11, 1))

with open("J_loc.dat") as fp:
    for line in fp:
        item = line.split()
        if line == "\n" or line.strip() == "":
            continue  # skip blank lines
        if item[0] == "#":
            a1 = int(item[1])
            a2 = int(item[2])
            i = int(item[3])
            j = int(item[4])
            k = int(item[5])
            J = float(item[6])
            J_r[a1, a2, i + 5, j + 5, k] = J


print("Done!")


from numpy import linalg as LA
import time

kmesh = (256, 256, 1)
num_kpoints = np.prod(kmesh)


@jit(nopython=True)
def fourier_transform(kmesh, J_r):
    num_kpoints = kmesh[0] * kmesh[1] * kmesh[2]

    J_q = np.zeros((num_kpoints, 3, 3), dtype=np.complex128)

    rec_vec = np.array(
        [
            [1.57119, 0.90712, 0.00000],
            [0.00000, 1.81425, 0.00000],
            [0.00000, 0.00000, 0.296097],
        ]
    )

    k_vecs = np.zeros((num_kpoints, 3))

    e = 0
    for i in range(kmesh[0]):
        for j in range(kmesh[1]):
            for k in range(kmesh[2]):
                for z in range(3):
                    k_vecs[e, z] = (
                        (rec_vec[0, z] * i / kmesh[0])
                        + (rec_vec[1, z] * j / kmesh[1])
                        + (rec_vec[2, z] * k / kmesh[2])
                    )

            e += 1

    cell_vec = np.array(
        [[3.999, 0.000, 0.00000], [-1.999, 3.463, 0.00000], [0.000, 0.000, 21.2200]]
    )

    for i in range(-5, 5):
        for j in range(-5, 5):
            r = i * cell_vec[0] + j * cell_vec[1]
            t = (k_vecs @ r).reshape(num_kpoints, 1, 1)
            t = np.exp(-1j * t)
            rhs = np.ascontiguousarray(J_r[:, :, i + 5, j + 5, 0])
            J_q += t * rhs

    J_q = (
        J_q + np.transpose(J_q.conj(), (0, 2, 1))
    ) / 2  # avoid small numerical error from J_r

    return J_q


def compute_Tc(kmesh, J_r):
    J_q = fourier_transform(kmesh, J_r)

    num_kpoints = kmesh[0] * kmesh[1] * kmesh[2]

    sia = 0.00071

    J00 = J_q[0, 0, 0] + J_q[0, 0, 1] + J_q[0, 0, 2] - sia
    J11 = J_q[0, 1, 0] + J_q[0, 1, 1] + J_q[0, 1, 2] - sia
    J22 = J_q[0, 2, 0] + J_q[0, 2, 1] + J_q[0, 2, 2] - sia

    I = np.array([[J00, 0, 0], [0, J11, 0], [0, 0, J22]], dtype=np.complex128)

    N_q = np.zeros((num_kpoints, 3, 3), dtype=np.complex128)
    w_q = np.zeros((num_kpoints, 3))

    for i in range(num_kpoints):
        N_q[i] = -(I - J_q[i])
        w, _ = LA.eigh(N_q[i])
        w_q[i] = 11604.525 * w  # from eV to K

    Tc_RPA = (2 / 3) * num_kpoints / np.sum(1 / (w_q))

    return Tc_RPA


import time

_ = compute_Tc(kmesh, J_r)
tick = time.time()
Tc_RPA = compute_Tc(kmesh, J_r)
tock = time.time()
print("RPA Tc:", Tc_RPA)
print("Time:", tock - tick)
