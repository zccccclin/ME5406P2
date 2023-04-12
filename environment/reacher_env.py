import os
import numpy as np
import pybullet as p
import time


from base_env import BaseEnv

class ReacherEnv(BaseEnv):
    def __init__(self, render=False, moving_goal=False, random_start=False, train=True, tolerance=0.02):
        super().__init__(render=render, train=train, tolerance=tolerance)
        self.moving_goal = moving_goal
        self.random_start = random_start

        # Pritn variables
        print("\033[92m {}\033[00m".format('\n----------Environment created----------'))
        print("\033[92m {}\033[00m".format(f"Moving target: {self.moving_goal}"))
        print("\033[92m {}\033[00m".format(f"Random start: {self.random_start}"))
        print("\033[92m {}\033[00m".format(f'Distance tolerance: {self.dist_tolerance}'))
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
        reward, dist, ori_err, done = self.compute_reward(obs[1], goal, act)

        self.ep_reward += reward
        self.ep_len += 1
        info = {"accumm_reward": self.ep_reward, 
                "accumm_steps": self.ep_len,
                "reward": reward,
                "dist": dist,
                "ori_err": ori_err}
        
        return obs, reward, done, info

    def compute_reward(self, ee_pos_ori, goal_pos_ori, action):
        ee_pos = ee_pos_ori[:3]
        ee_ori = ee_pos_ori[3:]
        goal_pos = goal_pos_ori[:3]
        goal_ori = goal_pos_ori[3:]
        dist = np.linalg.norm(ee_pos - goal_pos)
        # if len(self.cont_self) > 0 or len(self.cont_table) > 0:
        #     done = True
        #     reward = -10
        #     return reward, dist, done
        # sparse reward

        # Add orientation error
        ori_err = np.linalg.norm(ee_ori - goal_ori)
        if dist < self.dist_tolerance and ori_err < 0.03:
            done = True
            reward_dist = 1
        else:
            done = False
            reward_dist = -dist - ori_err
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

        return reward, dist, ori_err, done



