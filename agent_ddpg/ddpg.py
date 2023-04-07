import json
import os
import time
from collections import deque
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from memory import ReplayMemory
from model import Actor, Critic
from util import logger
from util.utilClass import ActionNoise
from util.utilFunc import w_decay

class DDPG_Agent:
    def __init__(self, env, args):
        # Environment variables
        self.env = env
        self.obs_space = env.observation_space
        self.act_space = env.action_space
        self.obs_dim = self.obs_space.shape[0]
        self.act_dim = self.act_space.shape[0]
        self.goal_dim = self.env.goal_dim
        
        # Hyperparameters
        self.num_iters = args.num_iters
        self.warmup_iter = args.warmup_iter
        self.random_action_prob = args.random_action_prob
        self.tau = args.tau
        self.gamma = args.gamma
        self.rollout_steps = args.rollout_steps
        self.batch_size = args.batch_size
        self.train_steps = args.train_steps
        self.memory_capacity = args.memory_capacity
        self.use_her = args.use_her
        self.k = args.k
        self.tolerance = args.tolerance
        self.test_case_num = args.test_case_num


        # Logging variables
        self.best_mean_dist = np.inf
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.env_dir = os.path.join(args.save_dir, args.env)
        self.save_dir = os.path.join(self.env_dir, 'models')
        os.makedirs(self.save_dir, exist_ok=True)
        self.load_dir = args.load_dir
        self.global_step = 0

        # Actor-Critic model
        self.actor = Actor(self.obs_dim, self.act_dim, args.hid1_dim, args.hid2_dim, args.hid3_dim)
        self.critic = Critic(self.obs_dim, self.act_dim, args.hid1_dim, args.hid2_dim, args.hid3_dim)
        if args.resume or args.test or args.load_dir is not None:
            self.load_model(args.resume_iter, args.load_dir)
        if not args.test:
            self.actor_target = Actor(self.obs_dim, self.act_dim, args.hid1_dim, args.hid2_dim, args.hid3_dim)
            self.critic_target = Critic(self.obs_dim, self.act_dim, args.hid1_dim, args.hid2_dim, args.hid3_dim)
            self.actor_optim = self.init_optim(self.actor, lr=args.actor_lr)
            self.critic_optim = self.init_optim(self.critic, lr=args.critic_lr, weight_decay=args.critic_weight_decay)
            self.hard_update(self.actor_target, self.actor)
            self.hard_update(self.critic_target, self.critic)
            self.actor_target.eval()
            self.critic_target.eval()

            # Initialize action noise
            self.action_noise = ActionNoise(-0., 0.5, 0.0)

            # Replay memory
            self.memory = ReplayMemory(self.memory_capacity, act_dim=self.act_dim, obs_dim=self.obs_dim, goal_dim=self.goal_dim)
            self.critic_loss = nn.MSELoss()

        # Enable CUDA
        self.critic.cuda()
        self.actor.cuda()
        if not args.test:
            self.critic_target.cuda()
            self.actor_target.cuda()
            self.critic_loss.cuda()

        # Print model information
        print("Agent initialized")
        print("HER enabled: ", self.use_her)
        print("Moving target enabled: ", self.env.moving_goal)
        print("Training test case num: ", self.test_case_num)

    def test(self, record=True):
        mean_dist, success_rate = self.rollout(record=record)
        return mean_dist, success_rate
    
    def train(self):
        self.actor.train()
        self.critic.train()
        starttime = time.time()
        iter_rew = deque(maxlen=1)
        iter_step = deque(maxlen=1)
        total_rollout_step = 0

        for iter in range(self.global_step, self.num_iters):
            eps_rew = 0
            eps_step = 0
            self.action_noise.reset()
            obs = self.env.reset()
            obs = obs[0]
            iter_actor_loss = []
            iter_critic_loss = []
            if self.use_her:
                eps_exp = {'obs': [], 'act': [], 'rew': [], 'next_obs': [], 'achieved_goal': [], 'done': []}

            for rollout_step in range(self.rollout_steps):
                total_rollout_step += 1
                if self.load_dir is None and iter < self.warmup_iter or np.random.rand() < self.random_action_prob:
                    action = np.random.uniform(-1., 1., self.act_dim).flatten()
                else:
                    action = self.policy(obs).flatten()
                next_obs, rew, done, info = self.env.step(action)
                achieved_goal = next_obs[1].copy()
                next_obs = next_obs[0].copy()
                eps_rew += rew
                eps_step += 1
                self.memory.append(obs, action, rew, next_obs, achieved_goal, done)
                if self.use_her:
                    eps_exp['obs'].append(obs)
                    eps_exp['act'].append(action)
                    eps_exp['rew'].append(rew)
                    eps_exp['next_obs'].append(next_obs)
                    eps_exp['achieved_goal'].append(achieved_goal)
                    eps_exp['done'].append(done)
                obs = next_obs
            
            iter_rew.append(eps_rew)
            iter_step.append(eps_step)

            # HindSight Experience Replay
            if self.use_her:
                for t in range(eps_step - self.k):
                    ob = eps_exp['obs'][t]
                    action = eps_exp['act'][t]
                    next_ob = eps_exp['next_obs'][t]
                    achieved_goal = eps_exp['achieved_goal'][t]
                    k_future = np.random.choice(np.arange(t + 1, eps_step), self.k - 1, replace=False)
                    k_future = np.concatenate([np.array([t]), k_future])
                    for i in k_future:
                        future_achieved_goal = eps_exp['achieved_goal'][i]
                        her_obs = np.concatenate((ob[:-self.goal_dim], future_achieved_goal), axis=0)
                        her_next_obs = np.concatenate((next_ob[:-self.goal_dim], future_achieved_goal), axis=0)
                        her_rew, _, done = self.env.compute_reward(achieved_goal.copy(), future_achieved_goal, action)
                        self.memory.append(her_obs, action, her_rew, her_next_obs, achieved_goal.copy(), done)

            self.global_step += 1
            if iter >= self.warmup_iter:
                for t_train in range(self.train_steps):
                    actor_loss, critic_loss = self.update()
                    iter_actor_loss.append(actor_loss)
                    iter_critic_loss.append(critic_loss)
            
            # Log saving
            if iter % self.log_interval == 0:
                time_now = time.time()
                log = {}
                log['iter'] = iter
                log['ro_steps'] = total_rollout_step
                log['return'] = np.mean([rew for rew in iter_rew])
                log['steps'] = np.mean([step for step in iter_step])
                if iter > self.warmup_iter:
                    log['actor_loss'] = np.mean(iter_actor_loss)
                    log['critic_loss'] = np.mean(iter_critic_loss)
                log['time_elapsed'] = time_now - starttime
                for k, v in log.items():
                    logger.logkv(k, v)
                logger.dumpkvs()

            # Model saving
            if (iter == 0 or iter >= self.warmup_iter) and iter % self.save_interval == 0 and logger.get_dir():
                mean_dist, success_rate = self.rollout()
                logger.logkv('iter', iter)
                logger.logkv('test/ro_steps', total_rollout_step)
                logger.logkv('test/mean_dist', mean_dist)
                logger.logkv('test/success_rate', success_rate)
                
                # train_mean_dist, train_success_rate = self.rollout(train=True)
                # logger.logkv('train/mean_dist', train_mean_dist)
                # logger.logkv('train/success_rate', train_success_rate)

                logger.dumpkvs()
                print(f"Mean distance: {round(mean_dist, 3)}, Success rate: {round(success_rate * 100, 2)}" )

                if mean_dist < self.best_mean_dist:
                    self.best_mean_dist = mean_dist
                    best = True
                    print('*********************************************')
                    print('saving model with closes mean dist')
                    print('*********************************************')
                else:
                    best = False
                self.save_model(best, step=self.global_step)

    def update(self, batch):
        for k, v in batch.items():
            batch[k] = torch.from_numpy(v)
        obs0_t = batch['obs0']
        obs1_t = batch['obs1']
        obs0 = Variable(obs0_t).float().cuda()
        with torch.no_grad():
            obs1_var = Variable(obs1_t).float().cuda()
        
        rew = Variable(batch['rew']).float().cuda()
        act = Variable(batch['act']).float().cuda()
        done = Variable(batch['done']).float().cuda()

        critic_q_val = self.critic(obs0, act)
        with torch.no_grad():
            target_act = self.actor_target(obs1_var)
            target_q_val = self.critic_target(obs1_var, target_act)
            target_q_label = rew
            target_q_label += (1 - done) * self.gamma * target_q_val
            target_q_label = target_q_label.detach()
        
        self.actor.zero_grad()
        self.critic.zero_grad()
        critic_loss = self.critic_loss(critic_q_val, target_q_label)
        critic_loss.backward()
        self.critic_optim.step()

        self.critic.zero_grad()
        self.actor.zero_grad()
        actor_act = self.actor(obs0)
        actor_q_val = self.critic(obs0, actor_act)
        actor_loss = -actor_q_val.mean()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        return actor_loss.cpu().data.numpy(), critic_loss.cpu().data.numpy()
    
    def rollout(self, train=False, record=False):
        test_cases = self.test_case_num
        done_num = 0
        dist_list = []
        eps_len = []
        for case in range(test_cases):
            obs = self.env.reset()
            for step in range(self.rollout_steps):
                obs = obs[0].copy()
                action = self.policy(obs, stochastic=False).flatten()
                obs, rew, done, info = self.env.step(action)
                if done:
                    done_num += 1
                    break
            if record:
                print('dist: ', info['dist'])
            dist_list.append(info['dist'])
            eps_len.append(step)
        mean_dist = np.mean(np.array(dist_list))
        success_rate = done_num / float(test_cases)
        if record:
            with open(self.env_dir + '/test_data.json', 'w') as f:
                json.dump(dist_list, f)
            print('Test cases result: ')
            print(f'Min dist: {round(np.min(np.array(dist_list)),3)}, Max dist: {round(np.max(np.array(dist_list)),3)}.')
            print(f'Mean dist: {round(mean_dist,3)}, Success rate: {round(success_rate,3)}.')
        return mean_dist, success_rate

    def load_model(self, step=None, load_dir=None):
        save_dir = self.save_dir
        if load_dir is not None:
            model_file = os.path.join(load_dir, 'best.pth')
        else:
            if step is None:
                model_file = os.path.join(save_dir, 'best.pth')
            else:
                model_file = os.path.join(save_dir, '{:08d}.pth'.format(step))
        if not os.path.exists(model_file):
            raise ValueError('Model file {} does not exist'.format(model_file))
        
        print("Loading model from {}".format(model_file))
        checkpoint = torch.load(model_file)
        if load_dir is not None:
            actor_dict = self.actor.state_dict()
            critic_dict = self.critic.state_dict()
            actor_load_dict = {k: v for k, v in checkpoint['actor_state_dict'].items() if k in actor_dict}
            critic_load_dict = {k: v for k, v in checkpoint['critic_state_dict'].items() if k in critic_dict}
            actor_dict.update(actor_load_dict)
            critic_dict.update(critic_load_dict)
            self.actor.load_state_dict(actor_dict)
            self.critic.load_state_dict(critic_dict)
            self.global_step = 0
        else:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.global_step = checkpoint['global_step']
        if step is None:
            print("Model is {}".format(checkpoint['model_num']))

        self.warmup_iter += self.global_step
        print("Loading complete")

    def save_model(self, best, step=None):
        if step is None:
            step = self.global_step
        model_file = os.path.join(self.save_dir, '{:08d}.pth'.format(step))
        data = {'model_num': step,
                'global_step': self.global_step,
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optim_state_dict': self.actor_optim.state_dict(),
                'critic_optim_state_dict': self.critic_optim.state_dict()}
        print("\033[93m {}\033[00m".format("Saving model: %s" % model_file))
        torch.save(data, model_file)
        if best:
            torch.save(data, os.path.join(self.save_dir, 'best.pth'))

    def policy(self, obs, stochastic=True):
        self.actor.eval()
        obs = Variable(torch.from_numpy(obs)).float().cuda().view(1, -1)
        action = self.actor(obs).cpu().data.numpy()
        if stochastic:
            action = self.action_noise(action)
        self.actor.train()
        return action
    
    def init_optim(self, actor_critic, lr, weight_decay=0):
        params = w_decay([actor_critic], weight_decay=weight_decay)
        optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        return optim        

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)