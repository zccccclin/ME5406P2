import os
import numpy as np
import gym
from gym import spaces
import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data

class BaseEnv:
    def __init__(self, render, train=True, moving_goal=False, tolerance=0.01):    
        # Initialize Pybullet
        self.pc = bc.BulletClient(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

        # Initialize scene
        self.reset_scene(None)
        
        # Environment params
        self.dist_tolerance = tolerance
        self.joint_indices = range(1, 7)
        self.goal_dim = 3
        
        self.obs_dim = self.get_obs()[0].size
        self.act_dim = p.getNumJoints(self.arm_id) - 1

        high = np.inf * np.ones(self.obs_dim)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = self.update_action_space()


    def step(self, action):
        pass

    def reset(self, arm_id=None):
        pass

    def reset_scene(self, goal_pos=None):
        p.resetSimulation()
        table_base_pos = [0.5, 0, 0]
        arm_base_pos = [0, 0, 0.62]
        goal_pos = goal_pos if goal_pos is not None else [0, 0, 0]
        p.loadURDF("plane.urdf")
        p.loadURDF("table/table.urdf", basePosition=table_base_pos)
        self.goal_id = p.loadURDF('../assets/goal.urdf', goal_pos)
        self.arm_id = p.loadURDF("xarm/xarm6_robot_white.urdf", basePosition=arm_base_pos, useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION)

    
    def update_action_space(self):        
        # Torque array
        action_bound = np.array([50, 50, 32, 32, 32, 20])
        action_space = spaces.Box(-action_bound, action_bound, dtype=np.float32)

        return action_space

    def scale_action(self, act):
        act_h = (self.action_space.high - self.action_space.low) / 2
        act_l = (self.action_space.high + self.action_space.low) / 2
        return act * act_h + act_l

    def get_obs(self):
        noise = 0.02
        joint_states = p.getJointStates(self.arm_id, self.joint_indices)
        qpos = np.array([j[0] for j in joint_states])
        # qpos += np.random.uniform(-noise, noise, 6)
        qvel = np.array([j[1] for j in joint_states])
        # qvel += np.random.uniform(-noise, noise, 6)
        obs = np.concatenate([qpos, qvel])

        goal_pos = np.array(p.getBasePositionAndOrientation(self.goal_id)[0])
        ee_pos = np.array(p.getLinkState(self.arm_id, 6)[0])

        obs = np.concatenate([obs, ee_pos, goal_pos])


        return (obs, ee_pos)


