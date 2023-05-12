import blenderproc as bproc

######################## IMPORTS ########################
import numpy as np
import mathutils
import os
from pathlib import Path
import json

######################## GLOBALS ########################
CONFIG = Path("config.json")
configs = json.load(open(CONFIG))
SCENE = configs.get("SCENE", 1)
DBG = bool(configs.get("DBG", 0))
N_FRAMES = configs.get("N_FRAMES", 1)
DATA_DIR: Path = Path(configs.get("DATA_DIR", "data"))
OUTPUT_DIR: Path = DATA_DIR / f"scene_{SCENE}"
#########################################################

if DBG:
    import debugpy
    import warnings

    warnings.warn("Waiting for debugger Attach...", UserWarning)
    debugpy.listen(5678)
    debugpy.wait_for_client()

scene_mapping = {
    0: "resources/haven/hdris/abandoned_tiled_room/abandoned_tiled_room_2k.hdr",
    1: "resources/haven/hdris/boiler_room/boiler_room_2k.hdr",
}

scene = scene_mapping[SCENE]

# 0. initialize blenderproc
bproc.init()

# 1.1 set scene
# TODO: (optionally) add ground for robot to stand on
bproc.world.set_world_background_hdr_img(scene)

# 1.2 set lighting source (location, energy level)
light00 = bproc.types.Light()
light00.set_type("POINT")
light00.set_location([5, -5, 5])
light00.set_energy(2000)

light01 = bproc.types.Light()
light01.set_type("POINT")
light01.set_location([5, 5, 5])
light01.set_energy(2000)

light02 = bproc.types.Light()
light02.set_type("POINT")
light02.set_location([-5, 5, 5])
light02.set_energy(2000)

light03 = bproc.types.Light()
light03.set_type("POINT")
light03.set_location([-5, -5, 5])
light03.set_energy(2000)

# 2. load robot
# NOTE: the urdf contains spot's body twice, because the base link, i.e. the one w/o parent, has to be removed.
#       now it's the first child of the base link and everything works properly
robot = bproc.loader.load_urdf(
    urdf_file="spot/spot_robot_simplified/spot_simplified.urdf"
)
robot.remove_link_by_index(index=0)
robot.set_ascending_category_ids()

# 2.1 analyse loaded robot a bit // textures!?
# # Scale 3D model from mm to m
# robot.set_scale([0.001, 0.001, 0.001])
# # Find all materials
# materials = bproc.material.collect_all()
# link_name_textures_list = [(link.get_name(), link.visuals) for link in robot.links]

# 3. set camera properties
# 3.1 set point of interest (all cam poses look towards it)
poi = np.array([0.0, 0.0, 0.0])

# 3.2 set camera trajectory properties
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
for i in range(N_FRAMES):
    # # Camera trajectory that defines a quater circle at constant height
    location_cam = np.array(
        [
            3 * np.cos(i / N_FRAMES * 2 * np.pi),
            5 * np.sin(i / N_FRAMES * 2 * np.pi),
            1,
        ]
    )

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
bproc.renderer.set_max_amount_of_samples(100)

# activate depth rendering
bproc.renderer.enable_depth_output(True)
# bproc.renderer.enable_segmentation_output(map_by=["instance", "name"])
# bproc.renderer.enable_normals_output()

# render the whole pipeline
data = bproc.renderer.render()

# save to different data formats
# write the data to a .hdf5 container
# bproc.writer.write_hdf5(f"{OUTPUT_DIR}/hdf5", data)
# write the animations into .gif files
# bproc.writer.write_gif_animation(OUTPUT_DIR, data, frame_duration_in_ms=80)

# save output in standard bop format
bproc.writer.write_bop(
    os.path.join(OUTPUT_DIR, "bop_data"),
    target_objects=robot.links,
    depths=data["depth"],
    colors=data["colors"],
    m2mm=False,
    calc_mask_info_coco=False,
)
