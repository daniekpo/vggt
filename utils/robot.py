import time

import rtde_control
import rtde_receive
import numpy as np

from utils.robot_gripper import RobotiqGripper

default_home_joint_pose = [
    -5.025327865277426,
    -1.4772643607905884,
    0.8192513624774378,
    -0.9127100271037598,
    4.7187347412109375,
    -0.3162348906146448,
]


def clip_pose(pose, cfg):
    pose[0] = np.clip(pose[0], cfg.robot.x_bounds[0], cfg.robot.x_bounds[1])
    pose[1] = np.clip(pose[1], cfg.robot.y_bounds[0], cfg.robot.y_bounds[1])
    pose[2] = np.clip(pose[2], cfg.robot.z_bounds[0], cfg.robot.z_bounds[1])
    return pose


class Robot:
    def __init__(
        self, host="169.254.129.1", home_pose_joint=default_home_joint_pose, tcp_bounds=None
    ):
        self.rtde_c = None
        self.rtde_r = None
        self.host = host
        self.gripper = RobotiqGripper()
        self.connect()
        self.home_pose_joint = home_pose_joint
        self.tcp_bounds = tcp_bounds

    def clip_pose(self, pose):
        if self.tcp_bounds is not None:
            pose[0] = np.clip(pose[0], self.tcp_bounds[0][0], self.tcp_bounds[0][1])
            pose[1] = np.clip(pose[1], self.tcp_bounds[1][0], self.tcp_bounds[1][1])
            pose[2] = np.clip(pose[2], self.tcp_bounds[2][0], self.tcp_bounds[2][1])
        return pose

    def connect(self):
        if self.rtde_c is None or not self.rtde_c.isConnected():
            self.rtde_c = rtde_control.RTDEControlInterface(self.host)

        if self.rtde_r is None or not self.rtde_r.isConnected():
            self.rtde_r = rtde_receive.RTDEReceiveInterface(self.host)

        self.gripper.connect(self.host, 63352)
        self.gripper.activate()

    def disconnect(self):
        if self.rtde_c is not None:
            self.rtde_c.disconnect()
        if self.rtde_r is not None:
            self.rtde_r.disconnect()

    def go_home(self, acc=0.2, vel=0.2, random_mutation=False, asynchronous=False):
        if self.home_pose_joint is None:
            raise ValueError("Home pose not set")
        if random_mutation:
            pose = self.home_pose_joint.copy()
            # randomly add or subtract 0.01 to 0.03 to the x,y,z of the home pose
            pose[0] += np.random.uniform(-0.03, 0.03)
            pose[1] += np.random.uniform(-0.03, 0.03)
            pose[2] += np.random.uniform(-0.03, 0.03)
        else:
            pose = self.home_pose_joint
        self.movej(pose, acc, vel, asynchronous)

    def is_gripper_open(self):
        return self.gripper.is_open()

    def is_moving(self):
        return not self.rtde_c.isSteady()

    def open_gripper(self):
        self.gripper.open()

    def close_gripper(self):
        self.gripper.close()

    def movej(self, pose, acc=0.1, vel=0.1, asynchronous=False):
        self.rtde_c.moveJ(pose, vel, acc, asynchronous)

    def movel(self, pose, acc=0.1, vel=0.1, asynchronous=False, use_safety_limits=True):
        if use_safety_limits and self.tcp_bounds is not None:
            pose = self.clip_pose(pose)
        self.rtde_c.moveL(pose, vel, acc, asynchronous)

    def movep(self, pose, acc=0.1, vel=0.1, asynchronous=False):
        self.rtde_c.movePath(pose, vel, acc, asynchronous)

    def stopl(self):
        self.rtde_c.stopL()

    def stopj(self):
        self.rtde_c.stopJ()

    def getl(self, include_timestamp=False):
        pose = self.rtde_r.getActualTCPPose()
        if include_timestamp:
            return pose, time.time()
        return pose

    def getj(self, include_timestamp=False):
        joints = self.rtde_r.getActualQ()
        if include_timestamp:
            return joints, time.time()
        return joints

    def close(self):
        self.disconnect()
        self.gripper.disconnect()
