Create by Ricky
Update date: 11/10/2023
  
After installing the F1tenth simulator (https://gitlab.engr.illinois.edu/GolfCar/f1tenth_simulator): 
1. Add "new.world" to the following repository: f1tenth_simulator/racecar_worlds/worlds/
2. Add "keyboard_teleop.py", "set_state.py", "way_pts.py" to the following repository: f1tenth_simulator/f1tenth-sim/scripts/
  
#### Launch environment
$ source devel/setup.bash  
$ roslaunch f1tenth-sim racecar.launch world_name:=new  
  
#### Run controller
$ source devel/setup.bash  
$ rosrun f1tenth-sim keyboard_teleop.py  
  
#### Reset racecar state
$ rosrun f1tenth-sim set_state.py  
