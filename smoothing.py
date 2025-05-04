import os
import numpy as np
import warp as wp
import warp.sim
import trimesh
import sys
import time
import json

import meshplot
from meshplot import plot, subplot, interact

meshplot.offline()

device = wp.get_device("cuda")
#print(f"Working on {device} device")

benchmark = True


@wp.kernel
def laplacian_smoothing_energy(mesh: wp.uint64,
                               V: wp.array(dtype=wp.vec3),
                               #energy_arr: wp.array(dtype=float),
                               energy: wp.array(dtype=float)):

    tid = wp.tid()

    f = tid

    v0_id = wp.mesh_get_index(mesh, 3*f+0)
    v1_id = wp.mesh_get_index(mesh, 3*f+1)
    v2_id = wp.mesh_get_index(mesh, 3*f+2)

    v0 = V[v0_id]
    v1 = V[v1_id]
    v2 = V[v2_id]

    l0 = wp.length_sq(v1 - v0)
    l1 = wp.length_sq(v2 - v1)
    l2 = wp.length_sq(v0 - v2)

    #energy_arr[3*f + 0] = l0
    #energy_arr[3*f + 1] = l1
    #energy_arr[3*f + 2] = l2
    wp.atomic_add(energy, 0, l0 + l1 + l2)


@wp.kernel
def take_step(V: wp.array(dtype=wp.vec3),
              g: wp.array(dtype=wp.vec3),
              learning_rate: float):

    tid = wp.tid()

    V[tid] = V[tid] - g[tid]*learning_rate


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python smoothing.py <path_to_obj_file>")
    else:
        obj_file = sys.argv[1]
        mesh = trimesh.load(obj_file, file_type='obj')

        V = np.array(mesh.vertices, dtype=np.float32)
        F = np.array(mesh.faces, dtype=np.int32)

        V_wp = wp.array(V, dtype=wp.vec3, requires_grad=True)
        F_wp = wp.array(F.flatten(), dtype=wp.int32)

        mesh_wp = wp.Mesh(points=V_wp, indices=F_wp)

        energy = wp.zeros(1, dtype=float, device=device, requires_grad=True)

        #energy_arr = wp.zeros((3*len(F_wp)), dtype=float,
        #                      device=device, requires_grad=True)

        num_iterations = 100
        learning_rate = 0.01/2.0

        start_time = time.time()

        for i in range(num_iterations):
            with wp.Tape() as tape:
                wp.launch(laplacian_smoothing_energy,
                          dim=len(F), inputs=[mesh_wp.id, V_wp, energy], device=device)
            wp.synchronize()

            #energy_sum = wp.utils.array_sum(energy_arr)

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
        entry = {
            "num_faces": int(F.shape[0]),
            "total_time_ms": round(elapsed_time_ms, 3),
            "num_iter": int(num_iterations)
            }

        print(f'"{os.path.basename(obj_file)}": {json.dumps(entry, indent=2)}')    
        print(",")   
        
        # print(
        #     f"Smoothing Warp: {elapsed_time_ms:.3f} ms, {elapsed_time_ms/num_iterations:.3f} ms per iteration")
