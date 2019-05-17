from numba import cuda
import numpy as np
from timeit import default_timer as timer


@cuda.jit('int64(int64, int64)', device=True)
def mul(a, b):
    return a * b


@cuda.jit('void(int64[:], int64[:], int64[:], int64)')
def matrix_mul_gpu(a, b, r, dim):
    x = (cuda.blockIdx.x * cuda.blockDim.x) + cuda.threadIdx.x
    y = (cuda.blockIdx.y * cuda.blockDim.y) + cuda.threadIdx.y

    if (x < dim) and (y < dim):
        r[(y * dim) + x] = 0

        for i in range(0, dim):
            r[(y * dim) + x] += mul(a[(y * dim) + i], b[(i * dim) + x])


def matrix_mul_cpu(a, b, r, dim):
    for x in range(0, dim):
        for y in range(0, dim):
            for i in range(0, dim):
                r[y * dim + x] += a[(y * dim) + i] * b[(i * dim) + x]


dim = 64
a = np.random.randint(150, size=dim**2).astype('int64')
b = np.random.randint(150, size=dim**2).astype('int64')
r = np.zeros(dim**2).astype('int64')

block = (32, 32)
grid = (int(dim/block[0]), int(dim/block[1]))

start = timer()
a_d = cuda.to_device(a)
b_d = cuda.to_device(b)
r_d = cuda.to_device(r)
matrix_mul_gpu[grid, block](a_d, b_d, r_d, dim)
cuda.synchronize()
a_d.to_host()
b_d.to_host()
result = r_d.copy_to_host()
dt = timer() - start

start2 = timer()
matrix_mul_cpu(a, b, r, dim)
dt2 = timer() - start2

if np.array_equal(result, r):
    print("Arrays are equal")

print("Matrix multiplied in gpu in %f s" % dt)
print("Matrix multiplied in cpu in %f s" % dt2)
