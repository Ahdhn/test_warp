import os
import numpy as np
import warp as wp
import warp.sim
import trimesh
import sys
import time

import meshplot
from meshplot import plot, subplot, interact

meshplot.offline()

device = wp.get_device("cuda")
print(f"Working on {device} device")

benchmark = True


@wp.kernel
def laplacian_smoothing_energy(E: wp.array(dtype=wp.int32),
                               V: wp.array(dtype=wp.vec3),
                               energy: wp.array(dtype=float)):

    tid = wp.tid()

    e = tid

    v0_id = E[2*e + 0]
    v1_id = E[2*e + 1]

    v0 = V[v0_id]
    v1 = V[v1_id]

    l = wp.length_sq(v1 - v0)

    wp.atomic_add(energy, 0, l)


@wp.kernel
def take_step(V: wp.array(dtype=wp.vec3),
              g: wp.array(dtype=wp.vec3),
              learning_rate: float):

    tid = wp.tid()

    V[tid] = V[tid] - g[tid]*learning_rate


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smoothing_ev.py <path_to_obj_file>")
    else:
        obj_file = sys.argv[1]
        mesh = trimesh.load(obj_file, file_type='obj')

        V = np.array(mesh.vertices, dtype=np.float32)
        F = np.array(mesh.faces, dtype=np.int32)
        
        print("#F = ", F.shape)

        E = np.concatenate([
            F[:, [0, 1]],
            F[:, [1, 2]],
            F[:, [2, 0]]
        ], axis=0)

        E = np.sort(E, axis=1)
        E = np.unique(E, axis=0)

        V_wp = wp.array(V, dtype=wp.vec3, requires_grad=True)
        E_wp = wp.array(E.flatten(), dtype=wp.int32)

        energy = wp.zeros(1, dtype=float, device=device, requires_grad=True)

        num_iterations = 100
        learning_rate = 0.01

        start_time = time.time()

        for i in range(num_iterations):
            with wp.Tape() as tape:
                wp.launch(laplacian_smoothing_energy,
                          dim=E.shape[0], inputs=[E_wp, V_wp, energy], device=device)
            wp.synchronize()

            tape.backward(energy)

            wp.launch(take_step, dim=len(V_wp), inputs=[
                      V_wp, V_wp.grad, learning_rate], device=device)
            wp.synchronize()

            if not benchmark:
                if i % 10 == 0:
                    V = V_wp.numpy().reshape(len(V_wp), 3)
                    plot(V, F, filename="mesh.html",
                         shading={"wireframe": True})
                    print(f"Iteration {i}: Energy = {energy.numpy()[0]}")

            tape.zero()
            energy.zero_()

        end_time = time.time()
        elapsed_time_ms = (end_time - start_time) * 1000
        print(
            f"Smoothing Warp: {elapsed_time_ms:.3f} ms, {elapsed_time_ms/num_iterations:.3f} ms per iteration")
