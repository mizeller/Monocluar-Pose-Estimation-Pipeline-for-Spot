# Spot x BlenderProc
## Creating some synthetic data

### 0. Preliminaries
Took CAD model and URDF file from this repo:
https://github.com/chvmp/spot_ros/tree/gazebo/spot_description

### 1. Setup - Reference Example from BlenderProc
```bash
blenderproc run examples/advanced/urdf_loading_and_manipulation/main.py examples/resources/medical_robot/miro.urdf examples/advanced/urdf_loading_and_manipulation/output
```
> The  `main.py` file contains the whole procedure of the scene setup (objects, cameras, lights, movements, etc)
> 
> `miro.urdf` is the robot model
> 
> `output dir`...self explanatory

