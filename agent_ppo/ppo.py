import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from model import ActorCritic


class PPO_Agent:
    def __init__(self, env):
        # Initialize hyperparam
        self.timestep_per_batch = 3000
        self.max_timesteps_per_episode = 1000
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.lr = 0.005

        # Get environment info
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # Intialize Actor Critic
        self.actor = ActorCritic("actor", self.obs_dim, self.act_dim, 64, 64)
        self.critic = ActorCritic("critic", self.obs_dim, 1, 64, 64)

        # Initialize optimizer
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # Covariance matrix
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        # Logging
        self.logger = {
            "dt" : time.time_ns(),
            "timestep" : 0,
            "eps" : 0,
            "batch_len" : [],
            "batch_reward" : [],
            "actor_loss" : [],
        }

    def learn(self, max_eps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timestep_per_batch} timesteps per batch for a total of {max_eps} episodes")
        step = 0
        eps = 0
        while eps < max_eps:
            batch_obs, batch_act, batch_log_prob, batch_rtg, batch_len = self.rollout()
            
            step += np.sum(batch_len)
            eps += 1
            self.logger["timestep"] = step
            self.logger["eps"] = eps

            V, _ = self.evaluate(batch_obs, batch_act)
            A_k = batch_rtg - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # Compute V and log prob
                V, curr_log_prob = self.evaluate(batch_obs, batch_act)
                # Compute ratio
                ratio = torch.exp(curr_log_prob - batch_log_prob)
                # Compute surrogate loss
                surr1 = ratio * A_k
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * A_k
                # Compute actor critic loss
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtg)

                # actor backprop
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                # critic backprop
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                # Logging
                self.logger["actor_loss"].append(actor_loss.detach())
            self._log_summary()

            if eps % 100 == 0:
                torch.save(self.actor.state_dict(), "./model/actor.pt")
                torch.save(self.critic.state_dict(), "./model/critic.pt")

    def rollout(self):
        batch_obs = []
        batch_act = []
        batch_log_prob = []
        batch_reward = []
        batch_len = []
        
        # Number of timestep in this batch
        t = 0
        while t < self.timestep_per_batch:
            ep_reward = []

            obs = self.env.reset()[0]
            done = False
            for ep_t in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)

                # Step
                act, log_prob = self.get_act(obs)
                obs, reward, done, _= self.env.step(act)
                obs = obs[0]
                ep_reward.append(reward)
                batch_act.append(act)
                batch_log_prob.append(log_prob)
                
                if done:
                    break
            # Collect length and rewards
            batch_len.append(ep_t + 1)
            batch_reward.append(ep_reward)

        # Compute return to go
        batch_rtg = self.compute_rtg(batch_reward)

        # Convert to tensor
        batch_obs = np.array(batch_obs)
        batch_act = np.array(batch_act)
        batch_log_prob = np.array(batch_log_prob)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_act = torch.tensor(batch_act, dtype=torch.float)
        batch_log_prob = torch.tensor(batch_log_prob, dtype=torch.float)

        self.logger["batch_len"] = batch_len
        self.logger["batch_reward"] = batch_reward
        return batch_obs, batch_act, batch_log_prob, batch_rtg, batch_len

    def get_act(self, obs):
        # Get action and log prob
        act_mean = self.actor(obs)
        dist = MultivariateNormal(act_mean, self.cov_mat)
        # get sample from distribution within bounds
        act = dist.sample()
        # ensure act is within bounds
        act = torch.clamp(act, -1, 1)

        log_prob = dist.log_prob(act)
        # print(act.detach().numpy())
        return act.detach().numpy(), log_prob.detach()

    def compute_rtg(self, batch_reward):
        batch_rtg = []
        for ep_reward in reversed(batch_reward):
            rtg = 0
            for reward in reversed(ep_reward):
                rtg = reward + self.gamma * rtg
                batch_rtg.insert(0, rtg)
        batch_rtg = torch.tensor(batch_rtg, dtype=torch.float)
        return batch_rtg

    def evaluate(self, batch_obs, batch_act):
        V = self.critic(batch_obs).squeeze()
        # Get log prob
        act_mean = self.actor(batch_obs)
        dist = MultivariateNormal(act_mean, self.cov_mat)
        log_prob = dist.log_prob(batch_act)

        return V, log_prob
    
    def _log_summary(self):
        delta_t = self.logger['dt']
        self.logger['dt'] = time.time_ns()
        delta_t = (self.logger['dt'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['timestep']
        i_so_far = self.logger['eps']
        avg_ep_lens = np.mean(self.logger['batch_len'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_reward']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_loss']])

		# Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

		# Reset batch-specific logging data
        self.logger['batch_len'] = []
        self.logger['batch_reward'] = []
        self.logger['actor_loss'] = []


import gym
import sys
sys.path.append("../environment")
from reacher_env import ReacherEnv
env = ReacherEnv(render=False, moving_goal=False, train=True, tolerance=0.02)
agent = PPO_Agent(env)
agent.learn(100000)