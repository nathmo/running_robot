# About

This repo contains code to control a running robot as part of my master thesis.

The main things it does are :
1) to teach me about Reinforcment learning
2) help size  / validate the motor specs for the robot
3) have a simple controller for walking on a threadmill with only the forward and vertical DOF free (no rotation of the robot base)
4) train a controller using RL for running on a threadmill with only the forward and vertical DOF free (no rotation of the robot base)
5) train a controller using RL for full untherered running with all the DOF free

To achieve that the scripts are organized such that there is a few shared file with the phyisical simulation, the NN model, the reward function.

we then have 3 high level script :
Train -> train the model, generate checkpoint in .onnx format
visualize -> simulate a policy on the virtual robot (URDF) in a 3D render
run -> run the .onnx policy and send the command to the motor.

## MJCF Debug Workflow

The URDF export is fully fixed, so the MuJoCo path now lives under `mujoco/`.

1) Generate the MJCF scaffold from the SolidWorks CSV:
`python mujoco/build_debug_model.py`

2) Open the live MuJoCo viewer with auto-reload:
`python viewer/mjcf_debug_viewer.py --build`

The generated model is a rigid-body scaffold with mesh visuals, inertial data, joint-anchor sites, and inactive equality placeholders. As you fill in the real 1-DOF pivot joints, save the XML and the viewer will reload it automatically.

## Manual MJCF Edits

Edit [mujoco/running_robot_debug.xml](mujoco/running_robot_debug.xml) directly when a joint needs to move.

To turn a placeholder into a real pivot, replace the comments inside the child body with a hinge joint like this:

```xml
<joint name="knee_right" type="hinge" pos="0 0 0" axis="0 1 0" limited="true" range="-1.0 0.8"/>
```

Use these rules:

1. `pos` is the pivot location in the child body frame. If it is wrong, move the body frame or move the joint site until the rotation happens around the right point.
2. `axis` is the hinge direction in the child body frame. If it is wrong, rotate the joint axis until the motion is aligned.
3. `range` is the min/max angle in radians. If the limits are wrong, edit the two numbers directly.
4. The green/red anchor sites in the XML are just debug markers. You can move them while tuning the joint location.
5. Save the file and keep the MuJoCo viewer open; it reloads the model automatically.

The current CSV export does not provide usable hinge axes or min/max joint angles, so those values have to be filled in by hand from the mechanism geometry.

