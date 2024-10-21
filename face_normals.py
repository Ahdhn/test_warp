import os
import numpy as np
import warp as wp
import warp.sim
import warp.render

import trimesh
import sys
import time

import meshplot
from meshplot import plot, subplot, interact

meshplot.offline()


@wp.kernel
def compute_face_normal(mesh: wp.uint64,
                        FN: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    f = tid

    # the face vertices
    FN[f] = wp.mesh_eval_face_normal(mesh, f)


@wp.kernel
def compute_vertex_normal(mesh: wp.uint64,
                          VN: wp.array(dtype=wp.float32)):
    tid = wp.tid()

    f = tid

    # the face three vertices
    fv = wp.vec3i(
        wp.mesh_get_index(mesh, 3*f+0),
        wp.mesh_get_index(mesh, 3*f+1),
        wp.mesh_get_index(mesh, 3*f+2)
    )

    # the face vertices
    normal = wp.mesh_eval_face_normal(mesh, f)

    for i in range(3):
        v = fv[i]
        for j in range(3):
            wp.atomic_add(VN, 3*v + j, normal[j])


def vertex_normal(obj_file):
    # load the mesh
    mesh = trimesh.load(obj_file, file_type='obj')

    V = np.array(mesh.vertices, dtype=np.float32)
    FV = np.array(mesh.faces, dtype=np.int32)

    V_wp = wp.array(V, dtype=wp.vec3)
    FV_wp = wp.array(FV.flatten(), dtype=wp.int32)

    mesh_wp = wp.Mesh(
        points=V_wp,
        indices=FV_wp)

    num_vertices = V.shape[0]
    VN = wp.empty(3*num_vertices, dtype=wp.float32)
    VN.zero_()

    start_time = time.time()
    wp.launch(compute_vertex_normal, dim=num_vertices, inputs=[mesh_wp.id, VN])

    # num_faces = FV.shape[0]
    # FN = wp.empty(num_faces, dtype=wp.vec3)
    # wp.launch(compute_face_normal, dim=num_faces, inputs=[mesh_wp.id, FN])

    wp.synchronize()
    end_time = time.time()

    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"Computation Time: {elapsed_time_ms:.3f} ms")

    VN_np = VN.numpy().reshape(num_vertices, 3)
    color = VN_np / (VN_np.sum(axis=1, keepdims=True) + 0.0001)
    color = (color + 1) / 2

    plot(mesh.vertices, mesh.faces, c=color,
         filename="mesh.html", shading={"wireframe": True})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_obj_file>")
    else:
        obj_file = sys.argv[1]
        vertex_normal(obj_file=obj_file)
