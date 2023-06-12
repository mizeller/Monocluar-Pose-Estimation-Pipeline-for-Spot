# 3DV Project - Monocular Pose Estimation for Human-Robot Co-Localization

## Project Description

### Main Contributions
- **Accurate NeRF model of Spot**
- **Modular Synthetic data pipeline for any robot/object using BlenderProc**
- **First Dataset of Spot in BOP format to train Pose Estimation Algorithms for Spot** (~ 2000 frames)
- **Conversion Scripts from BOP to slightly modified OnePose++ data**


## Installation
This project is quite modular and combines different pipelines/algorithms. Since these pipelines are very well documented, please follow the installation instructions of the respective projects.


### BlenderProc2
Blenderproc is required to create synthetic data. The installation instructions can be found [here](https://dlr-rm.github.io/BlenderProc/).
For this project, we additionally installed all HDRIs from polyhaven.com using `blenderproc download haven`.

### OnePose++
To install OnePose++, please follow the instructions from their ReadMe. We forked their repository and added some additional changes to make it compatible with our conversion scripts from BOP to their custom dataformat. The forked repository can be found [here](https://github.com/Maemaemaeko/OnePose_Plus_Plus_Spot) if the specific commits are of interest.


### Maplab
Maybe less important for this project, but still of interest for the robotics community, we tried used Maplab to generate the ground truth poses for the synthetic data. The installation instructions can be found [here](https://maplab.asl.ethz.ch/docs/master/index.html). Of course, this requires the user to have an environment with the different robots/agents/... . 


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
