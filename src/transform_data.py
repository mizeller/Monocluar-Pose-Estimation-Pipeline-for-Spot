from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import json


# more information about the BOP format can be found here:
# https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md


######################## GLOBALS ########################
global SCENE, DBG, N_FRAMES, INPUT_DIR, ONEPOSE_DATA
CONFIG = Path("config.json")
configs = json.load(open(CONFIG))
SCENE = configs.get("SCENE", 1)
DBG = bool(configs.get("DBG", 0))
N_FRAMES = configs.get("N_FRAMES", 1)
DATA_DIR: Path = Path(configs.get("DATA_DIR", "data"))
BOP_DATA: Path = DATA_DIR / f"scene_{SCENE}/bop_data"
ONEPOSE_DATA: Path = DATA_DIR / f"scene_{SCENE}/onepose_data"
MODEL: str = configs.get("MODEL", "nerf")
assert MODEL in ["nerf", "urdf"], "MODEL must be either 'nerf' or 'urdf'"
#########################################################


def get_ARPoses_txt():
    """modified ARPoses file structure"""
    # 1. load the camera intrinsics from the camera.json file
    scene_gt_json: Path = BOP_DATA / "train_pbr/000000/scene_gt.json"
    scene_gt_dict: dict = json.load(open(scene_gt_json))

    # 2. clean the scene_gt_dict

    # 2.1 remove all unnecessary objects
    for key in list(scene_gt_dict.keys()):
        scene_gt_dict[key] = scene_gt_dict[key][0]

    """
    at this point, scene_gt_dict is a dict[str, dict] with values like this:
    {
        "cam_R_m2c": [
            -0.00025947034009732306,
            0.9999995231628418,
            0.0009168935357593,
            0.3160759508609772,
            0.0009519258746877313,
            -0.948733389377594,
            -0.9487338662147522,
            4.362138861324638e-05,
            -0.31607601046562195,
        ],
        "cam_t_m2c": [
            -0.00013851519906893373,
            0.0005056734662503004,
            3.162277936935425,
        ],
        "obj_id": 1,
    }
    """

    # 2.2 convert the 3x3 rotation matrix to a quaternion
    for key, value in scene_gt_dict.items():
        rotation_array: np.ndarray = np.array(value["cam_R_m2c"])
        rotation_matrix: np.ndarray = rotation_array.reshape(3, 3)

        translation_array: np.ndarray = np.array(value["cam_t_m2c"])
        translation_matrix: np.ndarray = translation_array.reshape(3, 1)

        combined_matrix = np.concatenate((rotation_matrix, translation_matrix), axis=1)

        if DBG:
            print(rotation_matrix)
            print(translation_matrix)
            print(combined_matrix)  # these are T_wc matrices

        # rotation_matrix = np.linalg.inv(rotation_matrix)
        rotation_quaternion: np.ndarray = R.from_matrix(rotation_matrix).as_quat()

        # now overrwrite the rotation matrix with the quaternion
        # value["cam_R_m2c"] = rotation_quaternion.tolist()

    """
    at this point, the rotation matrix has been converted to a quaternion
    we can directly create the ARPoses.txt file from the information encoded in the scene_gt_dict
    """

    # 2. create the Frames.txt file
    arposes_txt: Path = ONEPOSE_DATA / "ARposes.txt"
    with open(arposes_txt, "w") as f:
        f.write(
            "# timestamp, tx, ty, tz, rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz\n"
        )
        for idx, line in enumerate(range(N_FRAMES)):
            values: dict = scene_gt_dict[str(idx)]
            tx: float = values["cam_t_m2c"][0]
            ty: float = values["cam_t_m2c"][1]
            tz: float = values["cam_t_m2c"][2]
            # qw: float = values["cam_R_m2c"][0]
            # qx: float = values["cam_R_m2c"][1]
            # qy: float = values["cam_R_m2c"][2]
            # qz: float = values["cam_R_m2c"][3]
            rxx: float = values["cam_R_m2c"][0]
            ryx: float = values["cam_R_m2c"][1]
            rzx: float = values["cam_R_m2c"][2]
            rxy: float = values["cam_R_m2c"][3]
            ryy: float = values["cam_R_m2c"][4]
            rzy: float = values["cam_R_m2c"][5]
            rxz: float = values["cam_R_m2c"][6]
            ryz: float = values["cam_R_m2c"][7]
            rzz: float = values["cam_R_m2c"][8]

            line = f"{idx/100}, {tx}, {ty}, {tz}, {rxx}, {ryx}, {rzx}, {rxy}, {ryy}, {rzy}, {rxz}, {ryz}, {rzz}\n"
            f.write(line)


def get_FRAMES_txt():
    """One Pose++ requires a FRAMES.txt file to be present in the data directory.
    It has the following format:

    # timestamp, frame_index, fx, fy, cx, cy
    ...
    """
    # 1. load the camera intrinsics from the camera.json file
    camera_json: Path = BOP_DATA / "camera.json"
    camera_dict: dict = json.load(open(camera_json))
    fx: float = camera_dict["fx"]
    fy: float = camera_dict["fy"]
    cx: float = camera_dict["cx"]
    cy: float = camera_dict["cy"]

    # add default intrinsics.txt file to the output directory (TODO)
    get_intrinsics_txt(fx, fy, cx, cy)

    # 2. create the Frames.txt file
    frames_txt: Path = ONEPOSE_DATA / "Frames.txt"
    with open(frames_txt, "w") as f:
        f.write("# timestamp, frame_index, fx, fy, cx, cy\n")
        for idx, line in enumerate(range(N_FRAMES)):
            line = f"{idx/100}, {idx}, {fx}, {fy}, {cx}, {cy}\n"
            f.write(line)


def get_box_txt(bbox_properties: dict):
    box_txt: Path = ONEPOSE_DATA / "Box.txt"
    with open(box_txt, "w") as file:
        file.write(
            "# {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(*bbox_properties.keys())
        )
        file.write(
            "{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(*bbox_properties.values())
        )


def get_intrinsics_txt(fx, fy, cx, cy):
    intrinsics_txt: Path = ONEPOSE_DATA / "intrinsics.txt"
    with open(intrinsics_txt, "w") as file:
        file.write(f"fx: {fx}\n")
        file.write(f"fy: {fy}\n")
        file.write(f"cx: {cx}\n")
        file.write(f"cy: {cy}\n")


def main():
    # 0. init empty output directory
    ONEPOSE_DATA.mkdir(parents=True, exist_ok=True)

    # 1. add default Box.txt file to the output directory
    # it contains the pose of the (constant) 3D bounding box of the object we want to train
    # OnePose++ for. It can be extracted directly from blender.
    # TODO: modify these values if necessary

    if MODEL == "nerf":
        bbox_properties = {
            # bounding box center coordinates in world coordinates
            "px": 0.05,  # urdf spot: 0.0,
            "py": 0.05,  # urdf spot: 0.0,
            "pz": 0.05,  # urdf spot: 0.0,
            # bounding box dimensions in world coordinates
            "ex": 0.2,  # urdf spot: 0.85,
            "ey": 0.82,  # urdf spot: 0.2,
            "ez": 0.32,  # urdf spot: 0.2,
            # bounding box orientation in world coordinates (quaternion)
            "qw": 0.0,
            "qx": 0.2,
            "qy": 0.0,
            "qz": 0.0,
        }
    elif MODEL == "urdf":
        bbox_properties = {
            # bounding box center coordinates in world coordinates
            "px": 0.0,
            "py": 0.0,
            "pz": 0.0,
            # bounding box dimensions in world coordinates
            "ex": 0.85,
            "ey": 0.2,
            "ez": 0.2,
            # bounding box orientation in world coordinates (quaternion)
            "qw": 0.0,
            "qx": 0.2,
            "qy": 0.0,
            "qz": 0.0,
        }

    get_box_txt(bbox_properties=bbox_properties)

    # 2. add FRAMES.txt file to the output directory. it contains the camera intrinsics for each frame
    # it calls get_intrinsics_txt() internally and creates this file as well
    get_FRAMES_txt()

    # 3. add ARposes.txt file to the output directory. it contains the camera poses for each frame
    get_ARPoses_txt()


if __name__ == "__main__":
    main()
