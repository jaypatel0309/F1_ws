# Lane Following using Pure Pursuit Controller on F1TENTH Car

Welcome! This is our final project for course ECE484-Principles-Of-Safe-Autonomy in 2023 Fall. The course page can be found [here](https://publish.illinois.edu/robotics-autonomy-resources/f1tenth/).

The project implements a vision-based lane following system. Our aim to make vehicle follow the lane accurately and quickly without collision using Pure Pursuit Controller given RGB images. Our vehicle platform is build on [F1TENTH](https://f1tenth.org/).

Please check out my [portfolio post](https://jackyyeh5111.github.io/lane-following-using-pure-pursuit-controller-on-f1tenth-car/) or our [final presentation video](https://www.youtube.com/watch?v=mselI6W_V-o) for a greater detailed description.

## Overview
The vehicle is able to follow the lane accurately without collision:
<figure style="border-style: none">
<img src="https://github.com/jackyyeh5111/jackyyeh5111.github.io/assets/22386566/7e11faf1-e84e-420c-ac01-fbc8b3902dc0">
</figure>

Lane detection result:  
<figure style="border-style: none">
<img width="200" alt="image" src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExdXltaXNscHg5d2tvemNubWNmZTVzZzJ4MWp2cnUwY242a3NqZG1iYyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/QL9o5rpbySvFbv40mc/giphy.gif">
</figure>

## Method
<img width="936" alt="image" src="https://github.com/jackyyeh5111/jackyyeh5111.github.io/assets/22386566/2c44461c-4c9d-469f-abe3-95bb0d005945">

The project built vision-based lane following system from scratch. Lane detector identifies the lane from the captured frame and provides imaginary waypoints candidates for the controller. Next, the controller selects the best waypoint based on the vehicle state, and sends out next control signal.

The whole system is integrated with ROS. It consists of three primary components:
1. Camera calibration
2. Lane detection
3. Controller

## Quick Starter Guide
Testing environment: Ubuntu 20.04 LTS

### Installation
1. Install ROS Noetic ([link](https://wiki.ros.org/noetic/Installation/Ubuntu))
2. Clone repo
    ```
    $ git clone https://github.com/jackyyeh5111/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final.git lane-follow-project
    $ cd lane-follow-project
    ```
3. Activate virtualenv and install dependencies
    ```
    $ pip install -r requirements.txt
    ```

### Usage

#### online usage
```python
$ cd src/
$ python3 lane_detector.py [params...]
```

important params:
- `--perspective_pts`: param for perpective projection. Four number represents respectively for src_leftx, src_rightx, laney, offsety.
- `--angle_limit`: limit vechicle turing angle.
- `--vel_min`: min velocity
- `--vel_max`: max velocity
- `--look_ahead`: fixed look ahead distance for pure pursuit controller. -1 denotes dynamic lookahead distance, which directly use the most distant waypoint as target point.
- `--obstacle_tolerate_dist`: car change velocity if obstacle is within this distance.

#### offline usage (for testing)
For offline, you have to prepare data folder beforehand. Two types of input data format is allowed. One is rosbag, the other option is to put sequential images under the same folder.

- Use rosbag (testing rosbag [download link](https://uofi.box.com/s/ivq5gv9ffxyqpugf4f5c0p13gmado4pe))
    ```
    $ python debug_helper.py --use_rosbag [params...] # Run program
    $ rosbag play <rosbag_path> # Run rosbag
    ```
    
- Sequential images (testing images [download link](https://uofi.box.com/s/82lk65dg8a9vkvc4hn17ffag5car7dva))
    ```
    $ python debug_helper.py -i <source dir> [-s <specified_image_id> -n <num_samples>]
    ```

## Simulation
Please refer to the folder [controller_simulation](https://github.com/jackyyeh5111/ECE484-Principles-Of-Safe-Autonomy-2023Fall-Final/tree/main/controller_simulation) for simulation.
![ezgif-7-0647951daf](https://github.com/jackyyeh5111/jackyyeh5111.github.io/assets/22386566/105990e7-43ca-422c-96f3-22af7c10cd99)

## Acknowledgement
My great team members:
- Chu-Lin Huang
- Huey-Chii Liang
- Jay Patel

And the support from Professor & TA of ECE484 course.