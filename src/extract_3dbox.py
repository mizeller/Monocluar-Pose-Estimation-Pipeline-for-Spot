import cv2
import numpy as np
from transforms3d import affines, quaternions
from pathlib import Path
from pyquaternion import Quaternion
import math
from pprint import pprint


# GLOBALS

# set global paths

# TODO: select correct data path
data_path: Path = Path("data/synthetic_data/scene_0/onepose")  # SPOT
# data_path: Path = Path("data/onepose_demo_data/demo_cam-annotate")  # DEMO DATA


image_file: Path = data_path / "0.png"
box_path = data_path / "Box.txt"
intrinsics_path = data_path / "intrinsics.txt"
AR_path = data_path / "ARposes.txt"
output_path: Path = Path("output/box3d.png")

# make sure these files exist
assert data_path.exists(), f"Data path {data_path} does not exist"
assert image_file.exists(), f"Image {image_file} does not exist"
assert box_path.exists(), f"Box file {box_path} does not exist"
assert (
    intrinsics_path.exists()
), f"Camera intrinsics file {intrinsics_path} does not exist"
assert AR_path.exists(), f"AR pose file {AR_path} does not exist"


def get_bbox3d():
    with open(box_path, "r") as f:
        lines = f.readlines()
    box_data = [float(e) for e in lines[1].strip().split(",")]
    px, py, pz = box_data[:3]
    ex, ey, ez = box_data[3:6]
    bbox_3d = (
        np.array(
            [
                [-ex, -ey, -ez],  # bottom back left
                [ex, -ey, -ez],  # back right bottom
                [ex, -ey, ez],  # top front left
                [-ex, -ey, ez],  # top back left
                [-ex, ey, -ez],  # bottom back right
                [ex, ey, -ez],  # bottom front right
                [ex, ey, ez],  # top front right
                [-ex, ey, ez],  # top back right
            ]
        )
        * 0.5
    )

    bbox_3d = np.insert(bbox_3d, 8, [px, py, py], axis=0)

    bbox_3d_homo = np.concatenate([bbox_3d, np.ones((len(bbox_3d), 1))], axis=1)
    return bbox_3d_homo


def data_process_anno():
    with open(intrinsics_path, "r") as f:
        content = f.read()
    lines = content.split("\n")
    fx = float(lines[0].split(": ")[1])
    fy = float(lines[1].split(": ")[1])
    cx = float(lines[2].split(": ")[1])
    cy = float(lines[3].split(": ")[1])
    K_homo = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]])
    return K_homo


def parse_box(i):
    with open(box_path, "r") as f:
        lines = f.readlines()
    data = [float(e) for e in lines[1].strip().split(",")]
    position = data[:3]
    quaternion = data[6:]

    # nice rotation around z
    # angle_z = 0.9 * math.pi  # 90 degrees in radians
    # axis_z = [0, 0, 1]
    # quaternion_z = Quaternion(axis=axis_z, radians=angle_z)

    # nice rotation around x
    # angle_x = 0 * math.pi  # 90 degrees in radians
    # axis_x = [1, 0, 0]
    # quaternion_x = Quaternion(axis=axis_x, radians=angle_x)

    # nice rotation around y
    # angle_y = (
    #     i * math.pi
    # )  # 90 degrees in radians # i = 1.04 together w/ rot around z works nicely
    # axis_y = [1, 1, 1]
    # quaternion_y = Quaternion(axis=axis_y, radians=angle_y)

    # quaternion = quaternion_y  # + quaternion_z
    # quaternion = quaternion.q

    rot_mat = quaternions.quat2mat(quaternion)

    T_ow = affines.compose(position, rot_mat, np.ones(3))
    return T_ow


def reproj(K_homo, pose, points3d_homo):
    assert K_homo.shape == (3, 4)
    assert pose.shape == (4, 4)
    assert points3d_homo.shape[0] == 4  # [4 ,n]

    reproj_points = K_homo @ pose @ points3d_homo
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points  # [n, 2]


def get_points(image, i):
    with open(AR_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]

    # eles = lines[frame+1].split(',')
    # data = [float(e) for e in eles]
    # position = data[1:4]
    # rot_mat = np.array(data[4:]).reshape(3, 3)
    # T_wc = affines.compose(position, rot_mat, np.ones(3))
    # T_oc = T_wc @ T_ow
    # _, bbox_3d_homo = get_bbox3d(box_path)
    # K_homo = data_process_anno(intrinsics)
    # reproj_box3d = reproj(K_homo, T_oc, bbox_3d_homo.T)

    eles = lines[1].split(",")
    data = [float(e) for e in eles]

    position = data[1:4]
    quaternion = data[4:]
    rot_mat = quaternions.quat2mat(quaternion)
    rot_mat = rot_mat @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T_ow = parse_box(i)
    T_cw = affines.compose(position, rot_mat, np.ones(3))
    T_wc = np.linalg.inv(T_cw)

    T_oc_original = T_wc @ T_ow  # orignal
    T_oc_edit = T_cw @ T_ow  # edit // changed T_wc to T_cw

    print("Original Transformation Matrix from Object to Camera")
    pprint(T_oc_original)

    print("Edited Transformation Matrix from Object to Camera")
    pprint(T_oc_edit)

    # use T_oc_edit for now, projects points correctly
    T_oc = T_oc_edit

    bbox_3d_homo = get_bbox3d()
    K_homo = data_process_anno()

    # print(K_homo)
    reproj_box3d = reproj(K_homo, T_oc, bbox_3d_homo.T)

    # just reproj the center and check where it lies
    # cv2.circle(image, (0, 0), 10, (255, 0, 0), -1)

    # print(reproj_box3d)
    draw_box3d(reproj_box3d=reproj_box3d, image=image)
    draw_world_frame(reproj_box3d, image)
    return


def draw_world_frame(reproj_box3d, image):
    # DBG: visualise separate corners first, then draw the lines
    origin = (int(reproj_box3d[8][0]), int(int(reproj_box3d[8][1])))
    pt_x = (int(reproj_box3d[8][0]) + 50, int(int(reproj_box3d[8][1])))
    pt_y = (int(reproj_box3d[8][0]), int(int(reproj_box3d[8][1])) + 50)

    image = cv2.circle(image, origin, 10, (0, 0, 0), -1)
    image = cv2.line(image, origin, pt_x, (0, 0, 255), 2)  # x-axis, red
    image = cv2.line(image, origin, pt_y, (0, 255, 0), 2)  # y-axis, green
    return


def draw_box3d(reproj_box3d, image):
    back_left_bottom = (int(reproj_box3d[0][0]), int(int(reproj_box3d[0][1])))
    back_right_bottom = (int(reproj_box3d[1][0]), int(int(reproj_box3d[1][1])))
    front_right_bottom = (int(reproj_box3d[2][0]), int(int(reproj_box3d[2][1])))
    front_left_bottom = (int(reproj_box3d[3][0]), int(int(reproj_box3d[3][1])))
    back_left_top = (int(reproj_box3d[4][0]), int(int(reproj_box3d[4][1])))
    back_right_top = (int(reproj_box3d[5][0]), int(int(reproj_box3d[5][1])))
    front_right_top = (int(reproj_box3d[6][0]), int(int(reproj_box3d[6][1])))
    front_left_top = (int(reproj_box3d[7][0]), int(int(reproj_box3d[7][1])))

    center = (int(reproj_box3d[8][0]), int(int(reproj_box3d[8][1])))

    # DBG: visualise separate corners first, then draw the lines
    image = cv2.circle(image, center, 10, (255, 0, 0), -1)

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


def main():
    # read the image from the dataset

    # for i in np.arange(0, 2, 0.01):
    i = 0
    image = cv2.imread(image_file._str)
    get_points(image, i)

    cv2.imwrite(
        str(output_path),
        image,
    )


if __name__ == "__main__":
    main()
