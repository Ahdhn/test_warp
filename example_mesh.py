# taken from warp/examples/core/example_mesh.py

import os
import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.render


@wp.kernel
def deform(positions: wp.array(dtype=wp.vec3), t: float):
    tid = wp.tid()

    x = positions[tid]

    offset = -wp.sign(x[0]*0.02)
    scale = wp.sin(t)
    x = x + wp.vec3(0.0, offset*scale, 0.0)

    positions[tid] = x


@wp.kernel
def simulation(positions: wp.array(dtype=wp.vec3),
               velocities: wp.array(dtype=wp.vec3),
               mesh: wp.uint64,
               margin: float,
               dt: float):

    tid = wp.tid()
    x = positions[tid]
    v = velocities[tid]

    v = v + wp.vec3(0.0, 0.0 - 9.8, 0.0) * dt - v * 0.1 * dt

    xpred = x+v*dt
    max_dist = 1.5

    query = wp.mesh_query_point_sign_normal(mesh, xpred, max_dist)
    if query.result:
        p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)

        delta = xpred - p

        dist = wp.length(delta)*query.sign
        err = dist - margin

        # mesh collision
        if err < 0.0:
            n = wp.normalize(delta)*query.sign
            xpred = xpred - n*err

    # pbh update
    v = (xpred - x) * (1.0/dt)
    x = xpred

    positions[tid] = x
    velocities[tid] = v


class Example:
    def __init__(self, stage_path="example_mesh.usd"):
        rng = np.random.default_rng(42)
        self.num_particles = 1000

        self.sim_dt = 1.0/60.0
        self.sim_time = 0.0
        self.sim_timer = {}
        self.sim_margin = 0.1

        usd_stage = Usd.Stage.Open(os.path.join(
            warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("bunny"))

        usd_scale = 10.0

        # create collision mesh
        self.mesh = wp.Mesh(
            points=wp.array(usd_geom.GetPointsAttr().Get()
                            * usd_scale, dtype=wp.vec3),
            indices=wp.array(
                usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
        )

        # rand particles
        init_pos = (rng.random((self.num_particles, 3)) -
                    np.array([0.5, -1.5, 0.5])) * 10.0
        init_vel = rng.random((self.num_particles, 3)) * 0.0

        self.positions = wp.from_numpy(init_pos, dtype=wp.vec3)
        self.velocities = wp.from_numpy(init_vel, dtype=wp.vec3)

        # renderer
        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None,
                        help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_mesh.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int,
                        default=500, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
