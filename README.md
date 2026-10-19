# Roboticarm_project
# Vision-Guided Robotic Pick-and-Place

A pick-and-place system built on the Viam platform during a September 2026 robotics
hackathon. A UFactory arm with a wrist-mounted Intel RealSense depth camera locates
blocks anywhere in its workspace, picks them up, and stacks them into a 3-2-1 pyramid.

## Demo

<img width="2048" height="1351" alt="1791819493634" src="https://github.com/user-attachments/assets/a4428e83-3a29-42c3-aced-1bac1a3ce11d" />


## What it does

The system takes an unstructured workspace with scattered blocks and:
1. Detects each block using 2D color detection from the RealSense camera.
2. Fuses those detections with depth data to compute a full 3D pose for each block.
3. Determines how to grasp each block based on its orientation.
4. Picks the blocks up one at a time and places them into a pyramid, computing each
   slot position and layer height automatically.

## How it works

**Perception.** Color detections from the RealSense camera are combined with the
camera's depth stream to turn each 2D detection into a 3D object pose, so the arm can
locate a block anywhere in the workspace rather than at fixed, pre-programmed positions.

**Grasp orientation.** For each block, the system parses the raw binary point cloud and
runs Principal Component Analysis (PCA) to find the block's principal axis. That axis is
then transformed through the robot's frame tree so the gripper approaches the block at
the correct angle.

**Placement.** Blocks are sequenced into a 3-2-1 pyramid using computed slot positions
and layer heights, so the stack is built from the geometry rather than hard-coded
coordinates.

**Closed-loop verification.** After each grasp, the system verifies whether the pick
succeeded and automatically retries if it did not, instead of blindly continuing.

## The hardest bug

My computed grasp angles were correct at 0 and 90 degrees but exactly 90 degrees wrong
at 45 degrees. The cause was that I was applying camera-frame angles as world rotations.
The error was invisible at the angles I happened to test first, which made it confusing
until I stopped trusting that the numbers "looked reasonable" and actually traced the
orientation through the frame tree. Fixing it meant genuinely understanding the
coordinate transforms rather than assuming they were right.

## Tech stack

- **Platform:** Viam
- **Language:** Python
- **Hardware:** UFactory arm, Intel RealSense depth camera
- **Key techniques:** 2D-to-3D pose estimation, point-cloud processing, PCA for grasp
  orientation, frame-tree transforms, closed-loop grasp verification

## Running it

1. Configure the arm and camera in Viam.
2. Install dependencies:
3. Update the code if needed
4. Run: build_pyramid.py

## Notes

Built in 2 days as a hackathon project. Worked as a solo project. Didn't win anything but got an appreciation letter and gift as a SO-101 robotic arm kit. 
