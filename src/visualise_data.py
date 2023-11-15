import cv2
import numpy as np
from transforms3d import affines
from pathlib import Path
import json


######################## GLOBALS ########################
global MODEL, SCENE, DBG, N_FRAMES, INPUT_DIR, ONEPOSE_DATA
CONFIG = Path("config.json")
configs = json.load(open(CONFIG))
SCENE = configs.get("SCENE", 1)
DBG = bool(configs.get("DBG", 0))
N_FRAMES = configs.get("N_FRAMES", 1)
N_Z_LVLS = configs.get("N_Z_LVLS", 1)  # how many different heights should be sampled
MODEL: str = configs.get("MODEL", "nerf")
DATA_DIR: Path = Path(configs.get("DATA_DIR", "data"))
BOP_DATA: Path = DATA_DIR / f"scene_{SCENE}-annotate/bop_data"
ONEPOSE_DATA: Path = DATA_DIR / f"scene_{SCENE}-annotate/onepose_data"
#########################################################


def get_bbox3d():
    with open(box_file, "r") as f:
        lines = f.readlines()
    box_data = [float(e) for e in lines[1].strip().split(",")]
    # add the center of the bounding box as well
    px, py, pz = box_data[:3]
    ex, ey, ez = box_data[3:6]

    # the dimensions corresond to the total depth, width and height of the box
    # we need half of these values to get the correct coordinates w.r.t. the center
    ex /= 2
    ey /= 2
    ez /= 2

    bbox_3d = np.array(
        [
            [px - ex, py - ey, pz - ez],  # back, left, down
            [px + ex, py - ey, pz - ez],  # front, left, down
            [px + ex, py - ey, pz + ez],  # front, left, up
            [px - ex, py - ey, pz + ez],  # back, left, up
            [px - ex, py + ey, pz - ez],  # back, right, down
            [px + ex, py + ey, pz - ez],  # front, right, down
            [px + ex, py + ey, pz + ez],  # front, right, up
            [px - ex, py + ey, pz + ez],  # back, right, up
        ]
    )

    # add the center of the bounding box as well for visualisation purposes
    bbox_3d = np.insert(bbox_3d, 8, [px, py, pz], axis=0)
    bbox_3d_homo = np.concatenate([bbox_3d, np.ones((len(bbox_3d), 1))], axis=1)

    return bbox_3d, bbox_3d_homo


def data_process_anno(intrinsics_path):
    with open(intrinsics_path, "r") as f:
        content = f.read()
    lines = content.split("\n")
    fx = float(lines[0].split(": ")[1])
    fy = float(lines[1].split(": ")[1])
    cx = float(lines[2].split(": ")[1])
    cy = float(lines[3].split(": ")[1])

    K_homo = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]])
    return K_homo


def reproj(K_homo, pose, points3d_homo):
    assert K_homo.shape == (3, 4)
    assert pose.shape == (4, 4)
    assert points3d_homo.shape[0] == 4  # [4 ,n]

    reproj_points = K_homo @ pose @ points3d_homo
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points  # [n, 2]


def draw_box3d(reproj_box3d, image):
    back_left_bottom = (
        int(reproj_box3d[0][0]),
        int(reproj_box3d[0][1]),
    )
    back_right_bottom = (
        int(reproj_box3d[1][0]),
        int(reproj_box3d[1][1]),
    )
    front_right_bottom = (
        int(reproj_box3d[2][0]),
        int(reproj_box3d[2][1]),
    )
    front_left_bottom = (
        int(reproj_box3d[3][0]),
        int(reproj_box3d[3][1]),
    )
    back_left_top = (
        int(reproj_box3d[4][0]),
        int(reproj_box3d[4][1]),
    )
    back_right_top = (
        int(reproj_box3d[5][0]),
        int(reproj_box3d[5][1]),
    )
    front_right_top = (
        int(reproj_box3d[6][0]),
        int(reproj_box3d[6][1]),
    )
    front_left_top = (
        int(reproj_box3d[7][0]),
        int(reproj_box3d[7][1]),
    )

    # center = (int(reproj_box3d[8][0]), int(int(reproj_box3d[8][1])))
    # # DBG: visualise separate corners first, then draw the lines
    # image = cv2.circle(image, center, 10, (255, 0, 0), -1)

    cv2.line(image, back_left_bottom, front_left_bottom, (0, 255, 0), 2)
    cv2.line(image, back_left_bottom, back_right_bottom, (0, 255, 0), 2)
    cv2.line(image, back_left_bottom, back_left_top, (0, 255, 0), 2)
    cv2.line(image, back_right_top, front_right_top, (0, 255, 0), 2)
    cv2.line(image, back_right_top, back_left_top, (0, 255, 0), 2)
    cv2.line(image, back_right_top, back_right_bottom, (0, 255, 0), 2)
    cv2.line(image, front_left_top, front_left_bottom, (0, 255, 0), 2)
    cv2.line(image, front_left_top, back_left_top, (0, 255, 0), 2)
    cv2.line(image, front_left_top, front_right_top, (0, 255, 0), 2)
    cv2.line(image, front_right_bottom, front_left_bottom, (0, 255, 0), 2)
    cv2.line(image, front_right_bottom, back_right_bottom, (0, 255, 0), 2)
    cv2.line(image, front_right_bottom, front_right_top, (0, 255, 0), 2)


def draw_world_frame(reproj_box3d, image):
    # DBG: visualise separate corners first, then draw the lines
    origin = (int(reproj_box3d[8][0]), int(int(reproj_box3d[8][1])))
    pt_x = (int(reproj_box3d[8][0]) + 50, int(int(reproj_box3d[8][1])))
    pt_y = (int(reproj_box3d[8][0]), int(int(reproj_box3d[8][1])) + 50)

    image = cv2.circle(image, origin, 10, (255, 255, 255), -1)
    image = cv2.line(image, origin, pt_x, (0, 0, 255), 2)  # x-axis, red
    image = cv2.line(image, origin, pt_y, (0, 255, 0), 2)  # y-axis, green
    return


def get_points(image, frame):
    T_ow = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    if MODEL == "poly":
        # NOTE - had to tweak T_ow to get the correct orientation...not sure why
        # that was the case for the poly model...
        theta = np.radians(45)  # Convert degrees to radians
        T_ow = np.array(
            [
                [np.cos(theta), 0, np.sin(theta), 0],
                [0, 1, 0, 0],
                [-np.sin(theta), 0, np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )

    with open(ar_file, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    eles = lines[frame + 1].split(",")
    data = [float(e) for e in eles]
    position = data[1:4]
    rot_mat = np.array(data[4:]).reshape(3, 3)
    T_wc = affines.compose(position, rot_mat, np.ones(3))
    T_oc = T_wc @ T_ow
    _, bbox_3d_homo = get_bbox3d()
    K_homo = data_process_anno(intrinsics_file)
    reproj_box3d = reproj(K_homo, T_oc, bbox_3d_homo.T)
    draw_box3d(reproj_box3d=reproj_box3d, image=image)
    # draw_world_frame(reproj_box3d=reproj_box3d, image=image)
    return 0


def main():
    global box_file, intrinsics_file, ar_file, output_path
    box_file = ONEPOSE_DATA / "Box.txt"
    intrinsics_file = ONEPOSE_DATA / "intrinsics.txt"
    ar_file = ONEPOSE_DATA / "ARposes.txt"
    output_path = ONEPOSE_DATA / "annotated_frames"
    output_path.mkdir(parents=True, exist_ok=True)

    # make sure these files exist, otherwise run python src/transform_data.py again
    assert ONEPOSE_DATA.exists(), f"Data path {ONEPOSE_DATA} does not exist"
    assert box_file.exists(), f"box_file file {box_file} does not exist"
    assert (
        intrinsics_file.exists()
    ), f"Camera intrinsics file {intrinsics_file} does not exist"
    assert ar_file.exists(), f"AR pose file {ar_file} does not exist"

    for frame in range(0, N_FRAMES * N_Z_LVLS):
        image_file: Path = BOP_DATA / f"train_pbr/000000/rgb/{str(frame).zfill(6)}.png"
        image = cv2.imread(str(image_file))

        get_points(image, frame)

        if DBG:
            print(f"Saving annotated frame to {output_path}/{str(frame).zfill(6)}.png")

        cv2.imwrite(
            str(output_path) + "/" + str(frame).zfill(6) + ".png",
            image,
        )


if __name__ == "__main__":
    main()
