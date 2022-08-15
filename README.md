# Simple Monocular Visual Odometry method in ROS package

## Introduction

![](./intro.gif)

- Mainly inspired by <https://github.com/avisingh599/mono-vo>
- Dataset: KITTI 00 sequence
- Extractor: FAST
- Tracker: LK Optical Flow
- Tested Environment: Ubuntu 20.04 LTS & ROS noetic.

## Requirement

- ROS kinetic or melodic or noetic [download](http://wiki.ros.org/ROS/Installation)
- KITTI odometry dataset [download](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

## Instruction

- Clone this repository `git clone https://github.com/SeungRyeol/simple_mono_vo_ros`.
- Put this direcotory into your workspace such as `catkin_ws`.
- Build the workspace using `catkin_make`.
- Update ROS packages using `rospack profile`.
- Edit the launch file `launch/mono_vo_ros.launch`
  - Change `fn_kitti` variable to the kitti path in your system.
- Run `roslaunch mono_vo_ros mono_vo_ros.launch`.
