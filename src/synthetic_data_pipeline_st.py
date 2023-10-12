import blenderproc as bproc
from blenderproc.python.loader.HavenEnvironmentLoader import (
    get_random_world_background_hdr_img_path_from_haven,
)

import numpy as np
import mathutils
import os
from pathlib import Path
import json
import random

######################## GLOBALS ########################
CONFIG = Path("config.json")
configs = json.load(open(CONFIG))
SCENE = configs.get("SCENE", 1)
DBG = bool(configs.get("DBG", 0))
RND_CAM = bool(configs.get("RND_CAM", 0))
N_FRAMES = int(configs.get("N_FRAMES", 1))
# how many different heights should be sampled
# for each z-level, N_FRAMES are generated
N_Z_LVLS = configs.get("N_Z_LVLS", 1)
DATA_DIR: Path = Path(configs.get("DATA_DIR", "data"))
OUTPUT_DIR: Path = DATA_DIR / f"scene_{SCENE}"
#########################################################


if DBG:
    import debugpy
    import warnings

    warnings.warn("Waiting for debugger Attach...", UserWarning)
    debugpy.listen(5678)
    debugpy.wait_for_client()

# init bproc & set scene
bproc.init()
bproc.world.set_world_background_hdr_img(
    get_random_world_background_hdr_img_path_from_haven("resources/haven/")
)

# load robot & set pose
robot = bproc.loader.load_obj(filepath="spot/nerf/nerf_spot.dae")
robot = robot[0]
robot.set_local2world_mat(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
)
robot.set_cp("category_id", 1)  # necessary for BOP writer
poi = np.array([0.0, 0.0, 0.0])  # point of interest


if RND_CAM:
    # Add translational random walk on top of the POI
    poi_drift = bproc.sampler.random_walk(
        total_length=N_FRAMES,
        dims=3,
        step_magnitude=0.0005,
        window_size=10,
        interval=[-0.003, 0.003],
        distribution="uniform",
    )

    # Rotational camera shaking as a random walk: Sample an axis angle representation
    camera_shaking_rot_angle = bproc.sampler.random_walk(
        total_length=N_FRAMES,
        dims=1,
        step_magnitude=np.pi / 64,
        window_size=10,
        interval=[-np.pi / 12, np.pi / 12],
        distribution="uniform",
        order=2,
    )

    camera_shaking_rot_axis = bproc.sampler.random_walk(
        total_length=N_FRAMES, dims=3, window_size=10, distribution="normal"
    )

    camera_shaking_rot_axis /= np.linalg.norm(
        camera_shaking_rot_axis, axis=1, keepdims=True
    )
x_offset = 1.0
y_offset = 1.0
# random initial position of camera
if RND_CAM:
    x_offset = random.uniform(2.5, 4.0)
    y_offset = random.uniform(2.5, 4.0)


for z in np.linspace(-0.5, 1.5, N_Z_LVLS):
    for i in range(N_FRAMES):
        x = x_offset * np.sin(i / (1 * N_FRAMES) * 2 * np.pi)
        y = y_offset * np.cos(i / (1 * N_FRAMES) * 2 * np.pi)
        z = z

        location_cam = np.array([x, y, z])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location_cam)

        if RND_CAM:
        # Compute rotation based on vector going from location towards poi + drift
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                poi + poi_drift[i] - location_cam
            )

            # random walk axis-angle -> rotation matrix
            R_rand = np.array(
                mathutils.Matrix.Rotation(
                    camera_shaking_rot_angle[i], 3, camera_shaking_rot_axis[i]
                )
            )

            # Add the random walk to the camera rotation
            rotation_matrix = R_rand @ rotation_matrix

        cam2world_matrix = bproc.math.build_transformation_mat(
            location_cam, rotation_matrix
        )

        bproc.camera.add_camera_pose(cam2world_matrix)

# render results & save to disk
bproc.renderer.set_max_amount_of_samples(30)
bproc.renderer.enable_depth_output(True)

data = bproc.renderer.render()
# bproc.writer.write_gif_animation(OUTPUT_DIR, data)
bproc.writer.write_bop(
    os.path.join(OUTPUT_DIR, "bop_data"),
    target_objects=[robot],
    depths=data["depth"],
    colors=data["colors"],
    m2mm=False,
    calc_mask_info_coco=False,
)
