Robot and frame:
- The arm's frame has its origin at the robot's base; x and y are horizontal and z is measured up from the table top: +x points forward, away from the base, +y to the robot's left, +z up.
- The arm's position is the point between the gripper's finger pads. With the gripper open, the lower edges of the pads are about 0.5 below it.

Gripper:
- Pointing straight down is roll 3.14 (or -3.14) and pitch 0; pointing down, yaw turns the gripper about the vertical, and at yaw 0 its fingers pinch along y. Yaws 3.14 apart pinch alike: of two such yaws, turn to the one nearer 0.
- Open, the fingers are about 8.3 cm apart. A positive width change opens them all the way, a negative one closes them until something stops them or they meet, and 0 leaves them as they are. A close lasts until an open.
- Closing swings the pads about 1.3 further down; a close also draws the gripper back by that much along its axis, so the fingers close where they were.

Cameras:
- agentview: fixed in front of the robot and to its right, looking across the table; +x points toward the upper right of the image, +y toward its left and +z toward its top.
- wrist: on the gripper, looking down past the fingers; with the gripper pointing down at yaw 0, +x points toward the bottom of the image and +y toward its right, and the spot straight below the fingers lies right of and above the image centre.

Ending:
- The episode ends by itself as soon as the task is done: while observations keep coming, the task is not done yet.
- The task has a time limit, and every action spends some of it.
