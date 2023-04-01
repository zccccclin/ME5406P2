import os
import numpy as np
import pybullet as p
import time

from base_env import BaseEnv

class ReacherEnv(BaseEnv):
    def __init__(self, render, train=True, moving_goal=False, tolerance=0.01):
        super().__init__(render=render, train=train, moving_goal=moving_goal, tolerance=tolerance)
        self.reset()
        self.joint_states = p.getJointStates(self.arm_id, self.joint_indices)

        
    def reset(self, goal_pos=None):
        if goal_pos is None:
            goal_pos = self.gen_goal()
        self.reset_scene(goal_pos)

        obs = self.get_obs
        self.ep_reward = 0
        self.ep_len = 0
        return obs

    def step(self, action):
        scaled_action = self.scale_action(action)
        ee_pos = np.array(p.getLinkState(self.arm_id, 6)[0])
        goal_pos = p.getBasePositionAndOrientation(self.goal_id)[0]
        step = (goal_pos - ee_pos) / 1000
        ll = [-3.142,-3.14,-3.142,-3.142,-3.142,-3.142]
        #upper limits for null space (todo: set them to proper range)
        ul = [3.142,0,3.142,3.142,3.142,3.142,3.142]
        #joint ranges for null space (todo: set them to proper range)
        desired_joint_positions = p.calculateInverseKinematics(self.arm_id, 6, goal_pos,[1,0,0,0],lowerLimits=ll, upperLimits=ul, residualThreshold=1e-5 )[:6]
        for idx, pos in zip(self.joint_indices,desired_joint_positions):
            p.setJointMotorControl2(
                bodyIndex=self.arm_id,
                jointIndex=idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=pos,
                #forces=torque
            )
        p.stepSimulation()

        obs = self.get_obs()
        goal_pos = np.array(p.getBasePositionAndOrientation(self.goal_id)[0])
        reward, dist, done = self.compute_reward(obs[1], goal_pos, action)

        self.ep_reward += reward
        self.ep_len += 1
        info = {"accumm_reward": self.ep_reward, 
                "accumm_steps": self.ep_len,
                "reward": reward,
                "dist": dist}
        
        return obs, reward, done, info

    def compute_reward(self, ee_pos, goal_pos, action):
        dist = np.linalg.norm(ee_pos - goal_pos)

        if dist < self.dist_tolerance:
            done = True
            reward_dist = 1
        else:
            done = False
            reward_dist = -1
        reward = reward_dist
        reward -= 0.1 * np.square(action).sum()
        return reward, dist, done

    def gen_goal(self):
        x = np.random.uniform(0.2,0.5)
        y = np.random.uniform(-0.5,0.5)
        z = np.random.uniform(0.63,0.5+0.62)
        return np.array([x,y,z])


def main():
    env = ReacherEnv(1)
    while True:
        action = np.random.uniform(-1., 1., env.act_dim)
        o,r,d,i = env.step(action)
        print(i)
        time.sleep(0.1)
    
    



if __name__ == "__main__":
    main()
