from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import json


# https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md


# GLOBALS
SCENE: int = 1
OUTPUT_DIR: Path = Path(f"spot/output_gt/scene_{SCENE}/bop_data")
NFRAMES: int = 100


def get_ARPoses_txt():
    # 1. load the camera intrinsics from the camera.json file
    scene_gt_json: Path = OUTPUT_DIR / "train_pbr" / "000000" / "scene_gt.json"
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
        rotation_quaternion: np.ndarray = R.from_matrix(rotation_matrix).as_quat()
        # now overrwrite the rotation matrix with the quaternion
        value["cam_R_m2c"] = rotation_quaternion.tolist()

    """
    at this point, the rotation matrix has been converted to a quaternion
    we can directly create the ARPoses.txt file from the information encoded in the scene_gt_dict
    """

    # 2. create the Frames.txt file
    onepose_dir: Path = OUTPUT_DIR / "onepose"
    onepose_dir.mkdir(parents=True, exist_ok=True)
    arposes_txt: Path = onepose_dir / "ARposes.txt"
    with open(arposes_txt, "w") as f:
        f.write("# timestamp, tx, ty, tz, qw, qx, qy, qz\n")
        for idx, line in enumerate(range(NFRAMES)):
            values: dict = scene_gt_dict[str(idx)]
            tx: float = values["cam_t_m2c"][0]
            ty: float = values["cam_t_m2c"][1]
            tz: float = values["cam_t_m2c"][2]
            qw: float = values["cam_R_m2c"][0]
            qx: float = values["cam_R_m2c"][1]
            qy: float = values["cam_R_m2c"][2]
            qz: float = values["cam_R_m2c"][3]

            line = f"{idx/100}, {tx}, {ty}, {tz}, {qw}, {qx}, {qy}, {qz}\n"
            f.write(line)


def get_FRAMES_txt():
    """One Pose++ requires a FRAMES.txt file to be present in the data directory.
    It has the following format:

    # timestamp, frame_index, fx, fy, cx, cy
    ...
    """
    # 1. load the camera intrinsics from the camera.json file
    camera_json: Path = OUTPUT_DIR / "camera.json"
    camera_dict: dict = json.load(open(camera_json))
    fx: float = camera_dict["fx"]
    fy: float = camera_dict["fy"]
    cx: float = camera_dict["cx"]
    cy: float = camera_dict["cy"]

    # 2. create the Frames.txt file
    onepose_dir: Path = OUTPUT_DIR / "onepose"
    onepose_dir.mkdir(parents=True, exist_ok=True)
    frames_txt: Path = onepose_dir / "Frames.txt"
    with open(frames_txt, "w") as f:
        f.write("# timestamp, frame_index, fx, fy, cx, cy\n")
        for idx, line in enumerate(range(NFRAMES)):
            line = f"{idx/100}, {idx}, {fx}, {fy}, {cx}, {cy}\n"
            f.write(line)


def main():
    get_FRAMES_txt()
    get_ARPoses_txt()


if __name__ == "__main__":
    main()
