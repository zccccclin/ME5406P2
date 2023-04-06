import os
import numpy as np
import pybullet as p
import time

from base_env import BaseEnv

class ReacherEnv(BaseEnv):
    def __init__(self, render=False, moving_goal=False, train=True, tolerance=0.02):
        super().__init__(render=render, train=train, tolerance=tolerance)
        self.moving_goal = moving_goal
        
    def reset(self, goal_pos=None):
        if self.moving_goal:
            goal_pos = self.gen_goal()
        else:
            goal_pos = np.array([0.5, 0.3, 0.7])
        self.reset_scene(goal_pos)

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

        obs = self.get_obs()
        goal_pos = np.array(p.getBasePositionAndOrientation(self.goal_id)[0])
        reward, dist, done = self.compute_reward(obs[1], goal_pos, act)

        self.ep_reward += reward
        self.ep_len += 1
        info = {"accumm_reward": self.ep_reward, 
                "accumm_steps": self.ep_len,
                "reward": reward,
                "dist": dist}
        
        return obs, reward, done, info

    def compute_reward(self, ee_pos, goal_pos, action):
        dist = np.linalg.norm(ee_pos - goal_pos)
        # sparse reward
        if dist < self.dist_tolerance:
            done = True
            reward_dist = 1
        else:
            done = False
            reward_dist = -dist 
        reward = reward_dist
        # Action penalty
        reward -= 0.01 * np.square(action).sum()

        # # dense reward
        # reward = -dist
        # if dist < self.dist_tolerance:
        #     reward = 1
        #     done = True
        # else:
        #     done = False

        return reward, dist, done

    def gen_goal(self):
        x = np.random.uniform(0.2,0.5)
        y = np.random.uniform(-0.5,0.5)
        z = np.random.uniform(0.63,0.5+0.62)
        return np.array([x,y,z])
    

