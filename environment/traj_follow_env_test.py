import os
import numpy as np
import pybullet as p
import time
from pyquaternion import Quaternion
from base_env import BaseEnv
from env_util.traj_gen import generate_ellipse_trajectory, generate_rectangular_trajectory
class TrajFollowEnv(BaseEnv):
    def __init__(self, render, train=True, random_traj=False, tolerance=0.05):
        super().__init__(render=render, train=train, tolerance=tolerance)
        self.random_traj = random_traj
        self.traj_step = 0
        self.in_range_step = 0
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.4])

        
    def reset(self):
        self.home_pos = np.array([0.21, 0, 0.8])
        self.reset_scene(home_pos=self.home_pos, goal_pos=self.home_pos, goal_size='trajfollow')
        self.traj_step = 0
        self.in_range_step = 0 
        if self.random_traj:
            self.gen_traj()
        else:
            self.traj = generate_ellipse_trajectory(self.home_pos, 0.2, 0.2, 400)
        self.traj = np.array(self.traj).T
        obs = self.get_obs
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
        
        # scaled_act = self.scale_action(act)
        # for jtn in self.joint_indices:
        #     p.setJointMotorControl2(self.arm_id, jtn, p.VELOCITY_CONTROL, force=0)
        #     p.setJointMotorControl2(self.arm_id, jtn, p.TORQUE_CONTROL, force=scaled_act[jtn-1])
        # p.stepSimulation()
        # if self.testing:
        #     time.sleep(0.05)

        # self.cont_table =p.getContactPoints(self.arm_id,self.table_id )
        # self.cont_self = p.getContactPoints(self.arm_id,self.arm_id)

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

    def compute_reward(self, ee_pose, goal_pose, action):
        ee_pos = np.array(ee_pose[:3])
        goal_pos = np.array(goal_pose[:3])
        ee_ori = ee_pose[3:]
        goal_ori = goal_pose[3:]
        dist = np.linalg.norm(ee_pos - goal_pos)

        if dist < self.dist_tolerance:
            done = False
            reward_dist = 0.1
            self.in_range_step += 1
        else:
            done = False
            reward_dist = -1
            self.in_range_step = 0
        if self.in_range_step > 30:
            done = True
            reward_dist = 1
            self.in_range_step = 0

        reward = reward_dist
        reward -= 0.1 * np.square(action).sum()
        return reward, dist, done

    def gen_traj(self):
        rand = np.random.randint(0,1)
        print(rand)
        a = np.random.uniform(0.1, 0.25)
        b = np.random.uniform(0.1, 0.25)
        if rand == 0:
            self.traj = generate_ellipse_trajectory(self.home_pos, a, b, 400)
        else:
            self.traj = generate_rectangular_trajectory(self.home_pos, a, b, 400)




def main():
    env = TrajFollowEnv(random_traj=True, render=True)
    env.reset()
    while True:
        action = np.random.uniform(-1., 1., env.action_space.shape[0])
        o,r,d,i = env.step(action)
        # print(env.traj_step, d, i['dist'], env.in_range_step)
        ee_orn = p.getLinkState(env.arm_id, 6)[1]
        goal_orn = p.getBasePositionAndOrientation(env.goal_id)[1]
        ee_q = Quaternion(ee_orn[3], ee_orn[0], ee_orn[1], ee_orn[2])
        goal_q = Quaternion(goal_orn[3], goal_orn[0], goal_orn[1], goal_orn[2])
        q_error = goal_q.inverse * ee_q
        print( q_error)
        # print(ee_q, p.getLinkState(env.arm_id, 6)[1])
        time.sleep(0.1)
    
    



if __name__ == "__main__":
    main()
