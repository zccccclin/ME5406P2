import os
import numpy as np
import gym
from gym import spaces
import pybullet as p
from pybullet_utils import bullet_client as bc
import pybullet_data

class BaseEnv:
    def __init__(self, render, train=True, tolerance=0.02, env_name='reacher'):    
        # Initialize Pybullet
        self.pc = bc.BulletClient(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.env_name = env_name
        
        if train:
            self.testing = False
        else:
            self.testing = True

        # Initialize scene
        self.reset_scene(None)
        
        # Environment params
        self.dist_tolerance = tolerance
        self.joint_indices = range(1, 7)
        if self.env_name == 'reacher' or self.env_name == 'trajfollow':
            self.goal_dim = 3
        else:
            self.goal_dim = 7
        self.obs_dim = self.get_obs()[0].size
        high = np.inf * np.ones(self.obs_dim)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def step(self, action):
        pass

    def reset(self, arm_id=None):
        pass

    def reset_scene(self, goal_pos=None, home_pos=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.4])
        table_base_pos = [0.5, 0, 0]
        arm_base_pos = [0, 0, 0.63]
        goal_pos = goal_pos if goal_pos is not None else [0, 0, 0]
        p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", basePosition=table_base_pos)
        if self.env_name == 'reacher' or self.env_name == 'reacher_pose':
            self.goal_id = p.loadURDF('../assets/goal.urdf', goal_pos)
        else:
            self.goal_id = p.loadURDF('../assets/goal_traj.urdf', goal_pos)
        p.resetBasePositionAndOrientation(self.goal_id,goal_pos,[1,0,0,0])
        self.arm_id = p.loadURDF("xarm/xarm6_robot_white.urdf", basePosition=arm_base_pos, useFixedBase=True,
                                 flags=p.URDF_USE_SELF_COLLISION)
        ll = [-3.142,-3.14,-3.142,-3.142,-3.142,-3.142]
        #upper limits for null space (todo: set them to proper range)
        ul = [3.142,0,3.142,3.142,3.142,3.142,3.142]
        #joint ranges for null space (todo: set them to proper range)
        
        if home_pos is not None:
            desired_joint_positions = p.calculateInverseKinematics(self.arm_id, 6, home_pos, [1,0,0,0], lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
            for idx, pos in zip(self.joint_indices,desired_joint_positions):
                p.resetJointState(self.arm_id, idx, pos)
        self.update_action_space()

    
    def update_action_space(self):        
        # Torque array for xarm6
        action_bound = np.array([50, 50, 32, 32, 32, 20])
        self.action_space = spaces.Box(-action_bound, action_bound, dtype=np.float32)

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
        goal_ori = np.array(p.getBasePositionAndOrientation(self.goal_id)[1])
        ee_pos = np.array(p.getLinkState(self.arm_id, 6)[0])
        ee_ori = np.array(p.getLinkState(self.arm_id, 6)[1])

        if self.env_name == 'reacher_pose':
            obs = np.concatenate([obs, ee_pos, ee_ori, goal_pos, goal_ori])
            ach_goal = np.concatenate([ee_pos, ee_ori])
        else:
            obs = np.concatenate([obs, ee_pos, goal_pos])
            ach_goal = ee_pos

        return obs, ach_goal

    def gen_goal(self):
        x = np.random.uniform(0.2,0.5)
        y = np.random.uniform(-0.5,0.5)
        z = np.random.uniform(0.7,1)
        return np.array([x,y,z])
    
