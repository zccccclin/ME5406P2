import os
import numpy as np
import pybullet as p
import time
from pyquaternion import Quaternion

from base_env import BaseEnv
from env_util.traj_gen import generate_ellipse_trajectory, generate_rectangular_trajectory

class TrajFollowEnv(BaseEnv):
    def __init__(self, render, train=True, random_traj=False, dist_tol=0.05, ori_tol=0.1, env_name='trajfollow', reward_type='dense'):
        super().__init__(render=render, train=train, dist_tol=dist_tol, ori_tol=ori_tol, env_name=env_name)
        self.random_traj = random_traj
        self.traj_step = 0
        self.in_range_step = 0
        self.reward_type = reward_type

        # Pritn variables
        print("\033[92m {}\033[00m".format('\n----------Environment created----------'))
        print("\033[92m {}\033[00m".format(f"Environment name: {self.env_name}"))
        print("\033[92m {}\033[00m".format(f"Random Trajectory: {self.random_traj}"))
        print("\033[92m {}\033[00m".format(f'Distance tolerance: {self.dist_tol}'))
        print("\033[92m {}\033[00m".format(f'reward_type: {self.reward_type}'))
        if self.env_name == 'trajfollow_pose':
            print("\033[92m {}\033[00m".format(f'Orientation tolerance: {self.ori_tol}'))
        print("\033[92m {}\033[00m".format(f'Observation dim: {self.obs_dim}'))
        print("\033[92m {}\033[00m".format(f'Goal dim: {self.goal_dim}'))
        print("\033[92m {}\033[00m".format('-----------------------------------------\n'))

    def reset(self):
        self.home_pos = np.array([0.21, 0, 0.8])
        self.reset_scene(home_pos=self.home_pos, goal_pos=self.home_pos)
        self.traj_step = 0
        self.in_range_step = 0 
        if self.random_traj:
            self.gen_traj()
        else:
            self.traj = generate_ellipse_trajectory(self.home_pos, 0.2, 0.2, 400)
        self.traj = np.array(self.traj).T
        obs = self.get_obs()
        self.ep_reward = 0
        self.ep_len = 0
        return obs

    def step(self, act):
        # Follow trajectory
        self.traj_step += 1
        p.resetBasePositionAndOrientation(self.goal_id,self.traj[self.traj_step],[1,0,0,0])
        if self.traj_step == self.traj.shape[0]-1:
            self.traj_step = 0

        # Action
        # ll = [-3.142,-3.14,-3.142,-3.142,-3.142,-3.142]
        # #upper limits for null space (todo: set them to proper range)
        # ul = [3.142,0,3.142,3.142,3.142,3.142,3.142]
        # #joint ranges for null space (todo: set them to proper range)
        # goal_pos = np.array(p.getBasePositionAndOrientation(self.goal_id)[0])
        # desired_joint_positions = p.calculateInverseKinematics(self.arm_id, 6, goal_pos,[1,0,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
        # for idx, pos in zip(self.joint_indices,desired_joint_positions):
        #         p.setJointMotorControl2(
        #             bodyIndex=self.arm_id,
        #             jointIndex=idx,
        #             controlMode=p.POSITION_CONTROL,
        #             targetPosition=pos,
        #             #forces=torque
        #         )
        # p.stepSimulation()
        
        scaled_act = self.scale_action(act)
        for jtn in self.joint_indices:
            p.setJointMotorControl2(self.arm_id, jtn, p.VELOCITY_CONTROL, force=0)
            p.setJointMotorControl2(self.arm_id, jtn, p.TORQUE_CONTROL, force=scaled_act[jtn-1])
        p.stepSimulation()
        if self.testing:
            time.sleep(0.05)

        # self.cont_table =p.getContactPoints(self.arm_id,self.table_id )
        # self.cont_self = p.getContactPoints(self.arm_id,self.arm_id)

        obs = self.get_obs()
        goal_pos = np.array(p.getBasePositionAndOrientation(self.goal_id)[0])
        goal_ori = np.array(p.getBasePositionAndOrientation(self.goal_id)[1])
        goal = np.concatenate([goal_pos, goal_ori])
        if self.env_name == 'trajfollow_pose':
            reward, dist, ori_err, done = self.compute_reward(obs[1], goal, act)
        else:
            reward, dist, done = self.compute_reward(obs[1], goal, act)
        self.ep_reward += reward
        self.ep_len += 1
        info = {"accumm_reward": self.ep_reward, 
                "accumm_steps": self.ep_len,
                "reward": reward,
                "dist": dist,
                "in_range_step": self.in_range_step}
        if self.env_name == 'trajfollow_pose':
            info["ori_err"] = ori_err
        
        return obs, reward, done, info

    def compute_reward(self, current, target, action):
        ee_pos = current[:3]
        goal_pos = target[:3]
        dist = np.linalg.norm(ee_pos - goal_pos)

         # Add orientation error
        if self.env_name == 'trajfollow_pose':
            ee_ori = current[3:]
            goal_ori = target[3:]
            ee_q = Quaternion(a=ee_ori[3], b=ee_ori[0], c=ee_ori[1], d=ee_ori[2])
            goal_q = Quaternion(a=goal_ori[3], b=goal_ori[0], c=goal_ori[1], d=goal_ori[2])
            ori_err = Quaternion.distance(ee_q, goal_q)

        if self.env_name == 'trajfollow_pose':
            if dist < self.dist_tol:
                done = False
                reward_dist = 0.1
                if ori_err < self.ori_tol:
                    reward_dist = 0.2
                self.in_range_step += 1
            else:
                done = False
                if self.reward_type == 'dense':
                    reward_dist = - dist - ori_err
                else:
                    reward_dist = -1
                self.in_range_step = 0
            if self.in_range_step > 190:
                done = True
                reward_dist = 1
        else:
            if dist < self.dist_tol:
                done = False
                reward_dist = 0.1
                self.in_range_step += 1
            else:
                done = False
                if self.reward_type == 'dense':
                    reward_dist = - dist
                else:
                    reward_dist = -1
                self.in_range_step = 0
            if self.in_range_step > 190:
                done = True
                reward_dist = 1

        reward = reward_dist
        reward -= 0.1 * np.square(action).sum()

        if self.env_name == 'trajfollow_pose':
            return reward, dist, ori_err, done
        return reward, dist, done
    
    def gen_traj(self):
        rand = np.random.uniform(0,1)
        a = np.random.uniform(0.1, 0.25)
        b = np.random.uniform(0.1, 0.25)
        if rand < 0.5:
            self.traj = generate_ellipse_trajectory(self.home_pos, a, b, 200)
        else:
            self.traj = generate_rectangular_trajectory(self.home_pos, a, b, 200)
