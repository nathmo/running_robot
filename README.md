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


