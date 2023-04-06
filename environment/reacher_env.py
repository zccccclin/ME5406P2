import os
import numpy as np
import pybullet as p
import time

from base_env import BaseEnv

class ReacherEnv(BaseEnv):
    def __init__(self, render=False, train=True, moving_goal=False, tolerance=0.02):
        super().__init__(render=render, train=train, moving_goal=moving_goal, tolerance=tolerance)
        self.reset()
        
    def reset(self, goal_pos=None):
        # if goal_pos is None:
        #     goal_pos = self.gen_goal()
        goal_pos = np.array([0.5, 0.2, 0.7])
        self.reset_scene(goal_pos)

        obs = self.get_obs()[0]
        self.ep_reward = 0
        self.ep_len = 0
        return obs

    def step(self, act):
        # js = p.getJointStates(self.arm_id, self.joint_indices)
        # qpos = np.array([j[0] for j in js])
        # joint_state_target = act*0.05 + qpos
        # for jtn in self.joint_indices:
        #     p.setJointMotorControl2(self.arm_id, jtn, p.POSITION_CONTROL, targetPosition=joint_state_target[jtn-1], force=10)
        
        act = self.scale_action(act)
        for jtn in self.joint_indices:
            p.setJointMotorControl2(self.arm_id, jtn, p.VELOCITY_CONTROL, force=0)
            p.setJointMotorControl2(self.arm_id, jtn, p.TORQUE_CONTROL, force=act[jtn-1])
        # p.setJointMotorControlArray(bodyIndex=self.arm_id,
        #                             jointIndices=self.joint_indices,
        #                             controlMode=p.VELOCITY_CONTROL,
        #                             forces=np.zeros(6))
        # p.setJointMotorControlArray(bodyIndex=self.arm_id,
        #                             jointIndices=self.joint_indices,
        #                             controlMode=p.TORQUE_CONTROL,
        #                             forces=action)
        p.stepSimulation()

        obs = self.get_obs()
        goal_pos = np.array(p.getBasePositionAndOrientation(self.goal_id)[0])
        reward, dist, done = self.compute_reward(obs[1], goal_pos, act)

        self.ep_reward += reward
        self.ep_len += 1
        info = {"accumm_reward": self.ep_reward, 
                "accumm_steps": self.ep_len,
                "reward": reward,
                "dist": dist}
        
        return obs[0], reward, done, info

    def compute_reward(self, ee_pos, goal_pos, action):
        dist = np.linalg.norm(ee_pos - goal_pos)

        # sparse reward
        # if dist < self.dist_tolerance:
        #     done = True
        #     reward_dist = 1
        # else:
        #     done = False
        #     reward_dist = -1 
        # reward = reward_dist
        # # Action penalty
        # reward -= 0.01 * np.square(action).sum()

        # dense reward
        reward = -dist
        if dist < self.dist_tolerance:
            reward = 1
            done = True
        else:
            done = False

        return reward, dist, done

    def gen_goal(self):
        x = np.random.uniform(0.2,0.5)
        y = np.random.uniform(-0.5,0.5)
        z = np.random.uniform(0.63,0.5+0.62)
        return np.array([x,y,z])
    
    def action_space_sample(self):
        act = np.random.uniform(-1, 1, self.act_dim)
        return act

