import blenderproc as bproc

######################## IMPORTS ########################
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
DATA_DIR: Path = Path(configs.get("DATA_DIR", "data"))
OUTPUT_DIR: Path = DATA_DIR / f"scene_{SCENE}"
MODEL: str = configs.get("MODEL", "nerf")
assert MODEL in ["nerf", "urdf"], "MODEL must be either 'nerf' or 'urdf'"
#########################################################

######################## DESCRIPTION ####################
"""
1. set scene
2. set lighting source (location, energy level)
3. load robot
4. set camera properties
    4.1 set point of interest (all cam poses look towards it)
    4.2 set camera trajectory properties
        4.2.1 Add translational random walk on top of the POI
        4.2.2 Rotational camera shaking as a random walk: Sample an axis angle representation
    4.3 loop over frames // sample camera poses
"""
#########################################################


if DBG:
    import debugpy
    import warnings

    warnings.warn("Waiting for debugger Attach...", UserWarning)
    debugpy.listen(5678)
    debugpy.wait_for_client()


# 0. initialize blenderproc
bproc.init()

# 1.1 set scene
root: Path = Path("/Users/mizeller/projects/BlenderProc/resources/haven/hdris")
scene_folder: Path = random.choice(list(root.iterdir()))
scene: Path = list(scene_folder.iterdir())[0]
assert scene.exists(), f"Scene {scene} does not exist"
if DBG:
    print(f"Placing SPOT in scene: {scene}")
bproc.world.set_world_background_hdr_img(str(scene))


# 2. load robot
if MODEL == "nerf":
    # NOTE: robot from NeRF model
    robot = bproc.loader.load_obj(filepath="spot/nerf/nerf_spot.dae")
    robot = robot[0]
    # Set pose of object via local-to-world transformation matrix
    robot.set_local2world_mat(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # Set category id which will be used in the BopWriter
    robot.set_cp("category_id", 1)

elif MODEL == "urdf":
    # NOTE: the urdf contains spot's body twice, because the base link, i.e. the one w/o parent, has to be removed.
    #       now it's the first child of the base link and everything works properly
    # NOTE: robot w/o arm
    robot = bproc.loader.load_urdf(urdf_file="spot/spot_basic.urdf")
    robot.remove_link_by_index(index=0)
    robot.set_ascending_category_ids()


# 3. set camera properties
# 3.1 set point of interest (all cam poses look towards it)
poi = np.array([0.0, 0.0, 0.0])

# 3.2 set camera trajectory properties

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

# 3.3 loop over frames // sample camera poses
x_offset = 1.0
y_offset = 1.0
z_offset = 1.0
# random initial position of camera
if RND_CAM:
    x_offset = random.uniform(0.5, 2.5)
    y_offset = random.uniform(0.5, 2.0)
    z_offset = random.uniform(0.5, 1.5)

"""
for i in range(N_FRAMES):
    # Sample random camera location above objects
    location = np.random.uniform([-5, -5, 0], [3, 3, 3])
    # Compute rotation based on vector going from location towards poi
    rotation_matrix = bproc.camera.rotation_from_forward_vec(
        poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854)
    )
    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
    bproc.camera.add_camera_pose(cam2world_matrix)
"""


for i in range(N_FRAMES):
    x = x_offset * np.sin(i / (1 * N_FRAMES) * 2 * np.pi)
    y = y_offset * np.cos(i / (1 * N_FRAMES) * 2 * np.pi)
    z = z_offset

    # Camera trajectory that defines a quater circle at constant height
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

    # Add homog cam pose based on location an rotation
    cam2world_matrix = bproc.math.build_transformation_mat(
        location_cam, rotation_matrix
    )
    bproc.camera.add_camera_pose(cam2world_matrix)


# Set max samples for quick rendering
bproc.renderer.set_max_amount_of_samples(30)

# activate depth rendering
bproc.renderer.enable_depth_output(True)
# bproc.renderer.enable_segmentation_output(map_by=["instance", "name"])
# bproc.renderer.enable_normals_output()

# render the whole pipeline
data = bproc.renderer.render()

# bproc.writer.write_gif_animation(OUTPUT_DIR, data)
# save output in standard bop format

if MODEL == "nerf":
    bproc.writer.write_bop(
        os.path.join(OUTPUT_DIR, "bop_data"),
        target_objects=[robot],
        depths=data["depth"],
        colors=data["colors"],
        m2mm=False,
        calc_mask_info_coco=False,
    )
elif MODEL == "urdf":
    bproc.writer.write_bop(
        os.path.join(OUTPUT_DIR, "bop_data"),
        target_objects=robot.links,
        depths=data["depth"],
        colors=data["colors"],
        m2mm=False,
        calc_mask_info_coco=False,
    )
