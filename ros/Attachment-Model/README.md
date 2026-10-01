# Attachment-Model

ROS catkin package for MiRo attachment relationship behavior modeling.

## Package Structure
```
Attachment-Model/
├── msg/                      # Message definitions
│   ├── Action.msg           # Controller actions
│   ├── Emotion.msg          # Emotional parameters  
│   └── RobotPub.msg         # Robot state
├── src/                      # Python modules
│   ├── utils.py             # Constants and configurations
│   ├── subscribers.py       # ROS topic subscribers
│   ├── actuators.py         # Robot actuator controls (tail, LEDs, audio)
│   ├── perception.py        # AprilTag and audio perception
│   ├── behaviors.py         # Movement, exploration, tag tracking
│   ├── child_node.py        # Child robot main node
│   ├── parent_node.py       # Parent robot main node
│   ├── central_control.py   # Attachment dynamics controller
│   └── plot_graph.py        # Data visualization
├── launch/
│   └── relationship.launch  # Main launch file
├── package.xml
├── CMakeLists.txt
└── README.md
```

## Module Descriptions

- **utils.py**: MiRo constants (audio, visual parameters, camera calibration)
- **subscribers.py**: ROS subscribers for odometry, range, actions, robot states, emotions
- **actuators.py**: Controllers for tail wagging, audio playback, LED colors, sound detection
- **perception.py**: AprilTag detection and audio frequency analysis
- **behaviors.py**: Movement control, exploration with obstacle avoidance, tag tracking
- **child_node.py**: Child robot implementing proximity-seeking attachment behavior
- **parent_node.py**: Parent robot implementing caregiving attachment behavior
- **central_control.py**: Central controller managing attachment dynamics and action selection

## Setup

1. Clone into `~/mdk/catkin_ws/src/`
2. Navigate to `~/mdk/catkin_ws/` 
3. Run `catkin clean && catkin build`
4. Run `source devel/setup.bash`

## Running

1. `robot_switch miro`
2. `roscore`
3. Launch your Gazebo world
4. `roslaunch Attachment-Model relationship.launch`
