import cv2
import numpy as np
from transforms3d import affines
from pathlib import Path


# TODO: select correct data path
data_path: Path = Path("data/synthetic_data/scene_1/onepose")  # SPOT
# data_path: Path = Path("data/onepose_demo_data/demo_cam-annotate")  # DEMO DATA


def get_bbox3d(box_path):
    with open(box_path, "r") as f:
        lines = f.readlines()
    box_data = [float(e) for e in lines[1].strip().split(",")]
    ex, ey, ez = box_data[3:6]
    bbox_3d = (
        np.array(
            [
                [-ex, -ey, -ez],  # back, left, down
                [ex, -ey, -ez],  # front, left, down
                [ex, -ey, ez],  # front, left, up
                [-ex, -ey, ez],  # back, left, up
                [-ex, ey, -ez],  # back, right, down
                [ex, ey, -ez],  # front, right, down
                [ex, ey, ez],  # front, right, up
                [-ex, ey, ez],  # back, right, up
            ]
        )
        * 0.5
    )
    bbox_3d_homo = np.concatenate([bbox_3d, np.ones((8, 1))], axis=1)
    # print(bbox_3d_homo)
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


def get_points(box_path, image, AR_path, frame):
    T_ow = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    with open(AR_path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    eles = lines[frame + 1].split(",")
    data = [float(e) for e in eles]
    position = data[1:4]
    rot_mat = np.array(data[4:]).reshape(3, 3)
    T_wc = affines.compose(position, rot_mat, np.ones(3))
    T_oc = T_wc @ T_ow
    _, bbox_3d_homo = get_bbox3d(box_path)
    K_homo = data_process_anno(intrinsics)
    reproj_box3d = reproj(K_homo, T_oc, bbox_3d_homo.T)

    x0 = reproj_box3d[0][0]  # back, left, down
    y0 = reproj_box3d[0][1]
    x1 = reproj_box3d[1][0]  # front, left, down
    y1 = reproj_box3d[1][1]
    x2 = reproj_box3d[2][0]  # front, left, up
    y2 = reproj_box3d[2][1]
    x3 = reproj_box3d[3][0]  # back, left, up
    y3 = reproj_box3d[3][1]
    x4 = reproj_box3d[4][0]  # back, right, down
    y4 = reproj_box3d[4][1]
    x5 = reproj_box3d[5][0]  # front, right, down
    y5 = reproj_box3d[5][1]
    x6 = reproj_box3d[6][0]  # front, right, up
    y6 = reproj_box3d[6][1]
    x7 = reproj_box3d[7][0]  # back, right, up
    y7 = reproj_box3d[7][1]

    image = cv2.line(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x0), int(y0)), (int(x3), int(y3)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x0), int(y0)), (int(x4), int(y4)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x1), int(y1)), (int(x5), int(y5)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x2), int(y2)), (int(x6), int(y6)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x2), int(y2)), (int(x3), int(y3)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x3), int(y3)), (int(x7), int(y7)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x4), int(y4)), (int(x5), int(y5)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x4), int(y4)), (int(x7), int(y7)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x5), int(y5)), (int(x6), int(y6)), (0, 255, 0), 2)
    image = cv2.line(image, (int(x6), int(y6)), (int(x7), int(y7)), (0, 255, 0), 2)

    return 0


def main():
    global image_file, Box, intrinsics, AR, output_path
    Box = data_path / "Box.txt"
    intrinsics = data_path / "intrinsics.txt"
    AR = data_path / "ARposes.txt"
    output_path = Path("output/")

    # make sure these files exist
    assert data_path.exists(), f"Data path {data_path} does not exist"
    assert Box.exists(), f"Box file {Box} does not exist"
    assert intrinsics.exists(), f"Camera intrinsics file {intrinsics} does not exist"
    assert AR.exists(), f"AR pose file {AR} does not exist"

    for frame in range(0, 81):
        image_file = f"data/synthetic_data/scene_1/bop_data/train_pbr/000000/rgb/{str(frame).zfill(6)}.png"
        image = cv2.imread(image_file)

        get_points(Box, image, AR, frame)

        cv2.imwrite(
            str(output_path) + "/" + str(frame).zfill(6) + ".png",
            image,
        )


if __name__ == "__main__":
    main()
