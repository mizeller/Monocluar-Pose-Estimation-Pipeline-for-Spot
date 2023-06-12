# 3DV Project - Synthetic Data Pipeline
> **Students:** Mae Yamaguchi, Jonas Bohn, Christopher Tibaldo, Michel Zeller
> **Supervisors:** ...

## Usage
To create some synthetic data, run the following command:
```bash
bash run.sh
```



## Project Description
In a multi-agent SLAM ... 


## Synthetic Data
> [insert description of the synthetic pipeline here]


1. used spot CAD model and URDF file from [this](https://github.com/chvmp/spot_ros/tree/gazebo/spot_description) repo
2. ...
3. ...

## Projecting Real World data to Hololens Camera frame
The world_to_camera_projection.ipynb Contains the code with which we attempted to project the SPOT position into the Hololens camera frame. To run it requires a .bag recording of the Hololens and Spot robots, as well as a vertex_poses_velocities_biases.csv containing the positions and orientations of Hololens and SPOt in world coordinate frame, generated with MapLab
