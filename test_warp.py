import warp as wp
import numpy as np

num_points = 20


@wp.func
def lookup(foos: wp.array(dtype=wp.uint32), index: int):
    return foos[index]


@wp.func
def multi_valued_func(a: wp.float32, b: wp.float32):
    return a+b, a-b, a*b, a/b


@wp.kernel
def test_multi_valued_func(d1: wp.array(dtype=wp.float32), d2: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    d11, d22 = d1[tid], d2[tid]
    a, b, c, d = multi_valued_func(d11, d22)


@wp.struct
class mystruc:
    pos: wp.vec3
    vel: wp.vec3
    active: int
    indices: wp.array(dtype=int)


@ wp.kernel
def compute_length(points: wp.array(dtype=wp.vec3), lenghts: wp.array(dtype=float)):
    # thread index
    tid = wp.tid()

    # compute distance of each point from origin
    lenghts[tid] = wp.length(points[tid])


# allocate an array of 3d points
points = wp.array(np.random.rand(num_points, 3), dtype=wp.vec3)
lengths = wp.zeros(num_points, dtype=float)

# launch kernel
wp.launch(kernel=compute_length, dim=len(points),
          inputs=[points, lengths], device="cuda")

# allocate an uninit array of vec3s
v = wp.empty(shape=num_points, dtype=wp.vec3, device="cuda")

# allocate a zero init array of quaternions
q = wp.zeros(shape=num_points, dtype=wp.quat, device="cuda")

# from numpy
a = np.ones((10, 3), dtype=np.float32)
v = wp.from_numpy(a, dtype=wp.vec3, device="cuda")

print(lengths)
