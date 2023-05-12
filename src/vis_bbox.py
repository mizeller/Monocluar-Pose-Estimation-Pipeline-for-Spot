import cv2
import numpy as np
from pathlib import Path
from typing import Dict
from scipy.spatial.transform import Rotation as R


def main():
    # load onepose sample image
    img_pth: Path = Path("data/onepose_demo_data/demo_cam-annotate/Snapshot.png")
    assert img_pth.exists(), f"Image {img_pth} does not exist"
    img = cv2.imread(img_pth._str)

    # load Box.txt containing the ground truth (bounding box information and quaternion)
    gt_pth: Path = Path("data/onepose_demo_data/demo_cam-annotate/Box.txt")
    assert gt_pth.exists(), f"Ground truth {gt_pth} does not exist"

    # extract 2nd line of Box.txt containig relevant information
    data_tmp = open(gt_pth, "r").readlines()[1]
    data_tmp = data_tmp.split(",")
    data_keys = ["px", "py", "pz", "ex", "ey", "ez", "qw", "qx", "qy", "qz"]
    data_vals = [float(i) for i in data_tmp]
    gt: Dict[str:float] = {k: v for k, v in zip(data_keys, data_vals)}

    # Define the dimensions of the bounding box and the coordinate of one corner
    box_dims = (
        gt["ex"],
        gt["ey"],
        gt["ez"],
    )  # TODO: check order - depth, height, width

    # depth, height, width
    box_dims = [1, 10, 10]

    corner_coord = np.array(
        [gt["px"], gt["py"], gt["pz"]]
    )  # TODO: check order - x, y, z

    # Define the camera quaternion
    quat = [gt["qw"], gt["qx"], gt["qy"], gt["qz"]]  # w, x, y, z

    # Define the 3D coordinates of the bounding box
    # corners = np.array(
    #     [
    #         [corner_coord[0], corner_coord[1], corner_coord[2]],
    #         [corner_coord[0] + box_dims[2], corner_coord[1], corner_coord[2]],
    #         [
    #             corner_coord[0] + box_dims[2],
    #             corner_coord[1] + box_dims[1],
    #             corner_coord[2],
    #         ],
    #         [corner_coord[0], corner_coord[1] + box_dims[1], corner_coord[2]],
    #         [corner_coord[0], corner_coord[1], corner_coord[2] + box_dims[0]],
    #         [
    #             corner_coord[0] + box_dims[2],
    #             corner_coord[1],
    #             corner_coord[2] + box_dims[0],
    #         ],
    #         [
    #             corner_coord[0] + box_dims[2],
    #             corner_coord[1] + box_dims[1],
    #             corner_coord[2] + box_dims[0],
    #         ],
    #         [
    #             corner_coord[0],
    #             corner_coord[1] + box_dims[1],
    #             corner_coord[2] + box_dims[0],
    #         ],
    #     ]
    # )

    # Project the 3D bounding box onto the 2D image plane
    camera_matrix = np.eye(3)
    dist_coeffs = np.zeros((4, 1))

    rot = R.from_quat(quat)
    rot_matrix = rot.as_matrix()

    translation = np.array([300.0, 0.0, 0.0])
    # first component: left to right
    # second component: bottom to top

    imgpts, _ = cv2.projectPoints(
        corner_coord,
        rot_matrix,
        np.array([500.0, 300.0, 0.0]),
        camera_matrix,
        dist_coeffs,
    )

    # Draw the projected bounding box on the image
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # cv2.polylines(img, [imgpts[:4]], True, (0, 255, 0), 3)  # draw the front face
    # cv2.polylines(img, [imgpts[4:]], True, (0, 0, 255), 3)  # draw the back face
    for i, j in zip(range(4), range(4, 8)):
        cv2.line(
            img, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 3
        )  # draw the edges
    # Display the result
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
