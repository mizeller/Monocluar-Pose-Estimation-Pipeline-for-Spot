import blenderproc as bproc

"""
Synthetic data pipeline for Spot robot.
The pipeline consists of the following steps:
1.  define global parameters: N_FRAMES, DBG, idx, OUTPUT_DIR
    (idx is either 0 or 1, depending on which scene to use)
2.  initialize blenderproc
3.  load & set scene
4.  load & set lighting
"""

import numpy as np
import mathutils
import os
from pathlib import Path


# GLOBALS
N_FRAMES: int = 2
DBG: bool = True
scene_idx: int = 1  # 0 or 1, decide which scene to use
OUTPUT_DIR: Path = Path(f"output/scene_{scene_idx}")


if DBG:
    import debugpy
    import warnings

    warnings.warn("Waiting for debugger Attach...", UserWarning)
    debugpy.listen(5678)
    debugpy.wait_for_client()

scene_mapping = {
    1: "resources/haven/hdris/abandoned_tiled_room/abandoned_tiled_room_2k.hdr",
    2: "resources/haven/hdris/boiler_room/boiler_room_2k.hdr",
}

scene = scene_mapping[scene_idx]


bproc.init()


output_dir = f"spot/output/scene_{scene_idx}"

# set scene
# TODO: add ground for robot to stand on
bproc.world.set_world_background_hdr_img(scene)


# set lighting source (location, energy level)
def set_lighting():
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


set_lighting()

# load robot
# NOTE: I had to add the body twice, because the base link, i.e. the one w/o parent, had to be removed
#       and the body was initially the base link...now it's the first child of the base link and everything works properly
robot = bproc.loader.load_urdf(
    urdf_file="spot/spot_robot_simplified/spot_simplified.urdf"
)

robot.remove_link_by_index(index=0)
robot.set_ascending_category_ids()

# # Scale 3D model from mm to m
# robot.set_scale([0.001, 0.001, 0.001])

# body = robot.get_children()[2]
# Set category id which will be used in the BopWriter
# body.set_cp("category_id", 1)

# Find all materials
# materials = bproc.material.collect_all()

# analyse spot robot a bit
# link_name_textures_list = [(link.get_name(), link.visuals) for link in robot.links]

# set point of interest (all cam poses look towards it)
poi = np.array([0.0, 0.0, 0.0])

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
# render segmentation images
# bproc.renderer.enable_segmentation_output(map_by=["instance", "name"])
# bproc.renderer.enable_normals_output()

# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container

bproc.writer.write_hdf5(f"{output_dir}/hdf5", data)

# write the animations into .gif files
bproc.writer.write_gif_animation(output_dir, data, frame_duration_in_ms=80)

# save separate images
# write link poses in BOP format
bproc.writer.write_bop(
    os.path.join(output_dir, "bop_data"),
    target_objects=robot.links,
    depths=data["depth"],
    colors=data["colors"],
    m2mm=False,
    calc_mask_info_coco=False,
)
