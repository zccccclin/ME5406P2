import os
import numpy as np
import pybullet as p
import time
from pyquaternion import Quaternion


from base_env import BaseEnv

class ReacherEnv(BaseEnv):
    def __init__(self, render=False, moving_goal=False, random_start=False, train=True, dist_tol=0.02, ori_tol=0.1, env_name='reacher'):
        super().__init__(render=render, train=train, dist_tol=dist_tol, ori_tol=ori_tol, env_name=env_name)
        self.moving_goal = moving_goal
        self.random_start = random_start

        # Pritn variables
        print("\033[92m {}\033[00m".format('\n----------Environment created----------'))
        print("\033[92m {}\033[00m".format(f"Environment name: {self.env_name}"))
        print("\033[92m {}\033[00m".format(f"Moving target: {self.moving_goal}"))
        print("\033[92m {}\033[00m".format(f"Random start: {self.random_start}"))
        print("\033[92m {}\033[00m".format(f'Distance tolerance: {self.dist_tol}'))
        if self.env_name == 'reacher_pose':
            print("\033[92m {}\033[00m".format(f'Orientation tolerance: {self.ori_tol}'))
        print("\033[92m {}\033[00m".format(f'Observation dim: {self.obs_dim}'))
        print("\033[92m {}\033[00m".format(f'Goal dim: {self.goal_dim}'))
        print("\033[92m {}\033[00m".format('-----------------------------------------\n'))

    def reset(self, goal_pos=np.array([0.4, 0.3, 0.8])):
        if self.moving_goal:
            goal_pos = self.gen_goal()
        else:
            goal_pos = goal_pos
        if self.random_start:
            home_pos = self.gen_goal()
        else:
            home_pos = None
        self.reset_scene(goal_pos=goal_pos, home_pos=home_pos)

        obs = self.get_obs()
        self.ep_reward = 0
        self.ep_len = 0
        return obs

    def step(self, act):
        # js = p.getJointStates(self.arm_id, self.joint_indices)
        # qpos = np.array([j[0] for j in js])
        # joint_state_target = act*0.05 + qpos
        # for jtn in self.joint_indices:
        #     p.setJointMotorControl2(self.arm_id, jtn, p.POSITION_CONTROL, targetPosition=joint_state_target[jtn-1], force=10)
        
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
        if self.env_name == 'reacher_pose':
            reward, dist, ori_err, done = self.compute_reward(obs[1], goal, act)
        else:
            reward, dist, done = self.compute_reward(obs[1], goal, act)

        self.ep_reward += reward
        self.ep_len += 1
        info = {"accumm_reward": self.ep_reward, 
                "accumm_steps": self.ep_len,
                "reward": reward,
                "dist": dist}
        if self.env_name == 'reacher_pose':
            info['ori_err'] = ori_err
        
        return obs, reward, done, info

    def compute_reward(self, current, target, action):
        ee_pos = current[:3]
        goal_pos = target[:3]
        dist = np.linalg.norm(ee_pos - goal_pos)

         # Add orientation error
        if self.env_name == 'reacher_pose':
            ee_ori = current[3:]
            goal_ori = target[3:]
            ee_q = Quaternion(a=ee_ori[3], b=ee_ori[0], c=ee_ori[1], d=ee_ori[2])
            goal_q = Quaternion(a=goal_ori[3], b=goal_ori[0], c=goal_ori[1], d=goal_ori[2])
            ori_err = Quaternion.distance(ee_q, goal_q)
        
        # if len(self.cont_self) > 0 or len(self.cont_table) > 0:
        #     done = True
        #     reward = -10
        #     return reward, dist, done
        
        # Dense reward
        if self.env_name == 'reacher_pose':
            if dist < self.dist_tol and ori_err < self.ori_tol:
                done = True
                reward_dist = 1
            else:
                done = False
                reward_dist = -dist - ori_err
        else:
            if dist < self.dist_tol:
                done = True
                reward_dist = 1
            else:
                done = False
                reward_dist = -dist
        reward = reward_dist

        # Action penalty
        reward -= 0.01 * np.square(action).sum()

        # Without action penalty
        # reward = reward_dist
        if self.env_name == 'reacher_pose':
            return reward, dist, ori_err, done
        return reward, dist, done



