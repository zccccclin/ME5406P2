import json
import os
import time
from collections import deque

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

import mutils
from memory import Memory
from model import Actor, Critic
from mutils import OnlineMeanStd, safemean
from noise import OrnsteinUhlenbeckActionNoise, UniformNoise, NormalActionNoise
from util import logger

class DDPG:
    def __init__(self, env, args):
        # Environment variables
        self.ob_dim = ob_dim = env.observation_space.shape[0]
        self.ac_dim = ac_dim = env.action_space.shape[0]
        self.goal_dim = goal_dim = env.goal_dim
        self.env = env
        self.env_type = args.env
        
        # Hyperparameters
        self.num_iters = args.num_iters
        self.random_prob = args.random_prob
        self.tau = args.tau
        self.reward_scale = args.reward_scale
        self.gamma = args.gamma
        self.rollout_steps = args.rollout_steps
        self.batch_size = args.batch_size
        self.train_steps = args.train_steps
        self.warmup_iter = args.warmup_iter
        self.max_grad_norm = args.max_grad_norm
        self.use_her = args.use_her
        self.k_future = args.k_future
        self.test_case_num = args.test_case_num

        # Log and save params
        self.best_mean_err = np.inf
        self.best_mean_reward = -np.inf
        self.best_mean_irs = 0
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.env_dir = os.path.join(args.save_dir, args.env)
        self.model_dir = os.path.join(self.env_dir, 'model')
        self.load_dir = args.load_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.global_step = 0

        # Actor-Crtic model setup
        self.actor = Actor(ob_dim=ob_dim,
                           act_dim=ac_dim,
                           hid1_dim=args.hid1_dim,
                           hid2_dim=args.hid2_dim,
                           hid3_dim=args.hid3_dim,
                           init_method=args.init_method)
        self.critic = Critic(ob_dim=ob_dim,
                             act_dim=ac_dim,
                             hid1_dim=args.hid1_dim,
                             hid2_dim=args.hid2_dim,
                             hid3_dim=args.hid3_dim,
                             init_method=args.init_method)
        if args.resume or args.test or args.load_dir is not None:
            self.load_model(args.resume_step, load_dir=args.load_dir)
        if not args.test:
            self.actor_target = Actor(ob_dim=ob_dim,
                                      act_dim=ac_dim,
                                      hid1_dim=args.hid1_dim,
                                      hid2_dim=args.hid2_dim,
                                      hid3_dim=args.hid3_dim,
                                      init_method=args.init_method)
            self.critic_target = Critic(ob_dim=ob_dim,
                                        act_dim=ac_dim,
                                        hid1_dim=args.hid1_dim,
                                        hid2_dim=args.hid2_dim,
                                        hid3_dim=args.hid3_dim,
                                        init_method=args.init_method)
            self.actor_optim = self.construct_optim(self.actor,
                                                    lr=args.actor_lr)
            cri_w_decay = args.critic_weight_decay
            self.critic_optim = self.construct_optim(self.critic,
                                                     lr=args.critic_lr,
                                                     weight_decay=cri_w_decay)
            self.hard_update(self.actor_target, self.actor)
            self.hard_update(self.critic_target, self.critic)
            self.actor_target.eval()
            self.critic_target.eval()

            # Initialize action noise
            if args.noise_type == 'ou_noise':
                mu = np.zeros(ac_dim)
                sigma = float(args.ou_noise_std) * np.ones(ac_dim)
                self.action_noise = OrnsteinUhlenbeckActionNoise(mu=mu,
                                                                 sigma=sigma)
            elif args.noise_type == 'uniform':
                low_limit = args.uniform_noise_low
                high_limit = args.uniform_noise_high
                dec_step = args.max_noise_dec_step
                self.action_noise = UniformNoise(low_limit=low_limit,
                                                 high_limit=high_limit,
                                                 dec_step=dec_step)

            elif args.noise_type == 'gaussian':
                mu = np.zeros(ac_dim)
                sigma = args.normal_noise_std * np.ones(ac_dim)
                self.action_noise = NormalActionNoise(mu=mu,
                                                      sigma=sigma)

            # Replay memory
            self.memory = Memory(limit=int(args.memory_limit),
                                 action_shape=(int(ac_dim),),
                                 observation_shape=(int(ob_dim),),
                                 goal_shape=(int(goal_dim),))
            self.critic_loss = nn.MSELoss()
            self.ob_norm = args.ob_norm
            if self.ob_norm:
                self.obs_oms = OnlineMeanStd(shape=(1, ob_dim))
            else:
                self.obs_oms = None

        # Enable CUDA
        self.cuda()

        if not args.test:
            mutils.print_green('\n----------Agent initialized----------')
            mutils.print_green(f"HER enabled: {self.use_her}")
            mutils.print_green(f"Training test case num: {self.test_case_num}")
            mutils.print_green(f"Training epochs: {args.num_iters}")
            mutils.print_green('---------------------------------------')
        print('\n')

    def test(self, record=True):
        if self.env.env_name == 'trajfollow':
            dist, reward, succ_rate, irs = self.rollout(record=record)
            print('Final mean in range step: ', irs)
            print('Final mean distance: ', dist)
            print('Final mean reward: ', reward)
            print('Success percentage: ', succ_rate * 100, '%')
        elif self.env.env_name == 'trajfollow_pose':
            dist, ori_err, reward, succ_rate, irs = self.rollout(record=record)
            print('Final mean in range step: ', irs)
            print('Final mean distance: ', dist)
            print('Final mean orientation error: ', ori_err)
            print('Final mean reward: ', reward)
            print('Success percentage: ', succ_rate * 100, '%')
        elif self.env.env_name == 'reacher_pose':
            dist, ori_err, reward, succ_rate = self.rollout(record=record)
            print('Final mean distance: ', dist)
            print('Final mean orientation error: ', ori_err)
            print('Final mean reward: ', reward)
            print('Success percentage: ', succ_rate * 100, '%')
        elif self.env.env_name == 'reacher':
            dist, reward, succ_rate = self.rollout(record=record)
            print('Final mean distance: ', dist)
            print('Final mean reward: ', reward)
            print('Success percentage: ', succ_rate * 100, '%')
            

    def train(self):
        self.net_mode(train=True)
        tfirststart = time.time()
        epoch_episode_rewards = deque(maxlen=1)
        epoch_episode_steps = deque(maxlen=1)
        total_rollout_steps = 0

        for epoch in range(self.global_step, self.num_iters):
            episode_reward = 0
            episode_step = 0
            self.action_noise.reset()
            obs = self.env.reset()
            obs = obs[0]
            epoch_actor_losses = []
            epoch_critic_losses = []
            if self.use_her:
                ep_experi = {'obs': [], 'act': [], 'reward': [], 'new_obs': [], 'ach_goals': [], 'done': []}
            
            for t_rollout in range(self.rollout_steps):
                total_rollout_steps += 1
                rand = np.random.random(1)[0]
                if self.load_dir is None and epoch < self.warmup_iter or rand < self.random_prob:
                    act = self.random_action().flatten()
                else:
                    act = self.policy(obs).flatten()
                new_obs, r, done, info = self.env.step(act)
                ach_goals = new_obs[1].copy()
                new_obs = new_obs[0].copy()
                episode_reward += r
                episode_step += 1
                self.memory.append(obs, act, r * self.reward_scale, new_obs, ach_goals, done)
                if self.use_her:
                    ep_experi['obs'].append(obs)
                    ep_experi['act'].append(act)
                    ep_experi['reward'].append(r * self.reward_scale)
                    ep_experi['new_obs'].append(new_obs)
                    ep_experi['ach_goals'].append(ach_goals)
                    ep_experi['done'].append(done)
                if self.ob_norm:
                    self.obs_oms.update(new_obs)
                obs = new_obs

            epoch_episode_rewards.append(episode_reward)
            epoch_episode_steps.append(episode_step)
            # print(episode_reward, epoch)

            # Hindsight exp replay
            if self.use_her:
                for t in range(episode_step - self.k_future):
                    ob = ep_experi['obs'][t]
                    act = ep_experi['act'][t]
                    new_ob = ep_experi['new_obs'][t]
                    ach_goal = ep_experi['ach_goals'][t]
                    k_futures = np.random.choice(np.arange(t + 1, episode_step), self.k_future - 1, replace=False)
                    k_futures = np.concatenate((np.array([t]), k_futures))
                    for future in k_futures:
                        new_goal = ep_experi['ach_goals'][future]
                        her_ob = np.concatenate((ob[:-self.goal_dim], new_goal), axis=0)
                        her_new_ob = np.concatenate((new_ob[:-self.goal_dim], new_goal), axis=0)
                        res = self.env.compute_reward(ach_goal.copy(), new_goal, act)
                        if self.env.env_name == 'trajfollow' or self.env.env_name == 'reacher':
                            her_reward, dist, done = res
                        else:
                            her_reward, dist, ori_err, done = res
                        self.memory.append(her_ob, act, her_reward * self.reward_scale, her_new_ob, ach_goal.copy(), done)
            
            # Update
            self.global_step += 1
            if epoch >= self.warmup_iter:
                for t_train in range(self.train_steps):
                    act_loss, cri_loss = self.train_net()
                    epoch_critic_losses.append(cri_loss)
                    epoch_actor_losses.append(act_loss)

            # Log saving
            if epoch % self.log_interval == 0:
                tnow = time.time()
                stats = {}
                if self.ob_norm:
                    stats['ob_oms_mean'] = safemean(self.obs_oms.mean.numpy())
                    stats['ob_oms_std'] = safemean(self.obs_oms.std.numpy())
                stats['total_rollout_steps'] = total_rollout_steps
                stats['rollout_return'] = safemean([rew for rew in epoch_episode_rewards])
                stats['rollout_ep_steps'] = safemean([l for l in epoch_episode_steps])
                if epoch >= self.warmup_iter:
                    stats['actor_loss'] = np.mean(epoch_actor_losses)
                    stats['critic_loss'] = np.mean(epoch_critic_losses)
                stats['epoch'] = epoch
                stats['actor_lr'] = self.actor_optim.param_groups[0]['lr']
                stats['critic_lr'] = self.critic_optim.param_groups[0]['lr']
                stats['time_elapsed'] = tnow - tfirststart
                for name, value in stats.items():
                    logger.logkv(name, value)
                logger.dumpkvs()

            # Model saving
            if (epoch == 0 or epoch >= self.warmup_iter) and self.save_interval and epoch % self.save_interval == 0 and logger.get_dir():
                if self.env_type == 'trajfollow':
                    mean_final_dist, mean_final_reward, succ_rate, mean_irs= self.rollout()
                elif self.env_type == 'trajfollow_pose':
                    mean_final_dist, mean_final_ori_err, mean_final_reward, succ_rate, mean_irs = self.rollout()
                elif self.env_type == 'reacher_pose':
                    mean_final_dist, mean_final_ori_err, mean_final_reward, succ_rate = self.rollout()
                else:
                    mean_final_dist, mean_final_reward, succ_rate = self.rollout()
                logger.logkv('epoch', epoch)
                logger.logkv('total_rollout_steps', total_rollout_steps)
                logger.logkv('mean_final_dist', mean_final_dist)
                if self.env_type == 'reacher_pose' or self.env_type == 'trajfollow_pose':
                    logger.logkv('mean_final_ori_err', mean_final_ori_err)
                if self.env_type == 'trajfollow':
                    logger.logkv('mean_final_irs', mean_irs)
                logger.logkv('succ_rate', succ_rate)
                # self.log_model_weights()
                logger.dumpkvs()
                if self.env_type == 'trajfollow':
                    print(f"Mean in range step: {mean_irs}, Mean reward: {round(mean_final_reward, 3)}, Success rate: {round(succ_rate * 100, 2)}" )
                elif self.env_type == 'trajfollow_pose':
                    print(f"Mean in range step: {mean_irs}, Mean distance: {mean_final_dist}, Mean ori_err: {mean_final_ori_err}, Total err: {mean_final_dist + mean_final_ori_err}, Mean reward: {round(mean_final_reward, 3)}, Success rate: {round(succ_rate * 100, 2)}" )
                elif self.env_type == 'reacher_pose':
                    print(f"Mean distance: {mean_final_dist}, Mean ori_err: {mean_final_ori_err}, Total err: {mean_final_dist + mean_final_ori_err}, Mean reward: {round(mean_final_reward, 3)}, Success rate: {round(succ_rate * 100, 2)}" )
                else:
                    print(f"Mean distance: {mean_final_dist}, Mean reward: {round(mean_final_reward, 3)}, Success rate: {round(succ_rate * 100, 2)}" )
                # Update best model by closest distance or reward
                if self.env_type == 'trajfollow' or self.env_type == 'trajfollow_pose':
                    if mean_final_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_final_reward
                        is_best = True
                        print('*********************************************')
                        print('saving model with best mean reward')
                        print('*********************************************')
                    else:
                        is_best = False
                    self.save_model(is_best=is_best, step=self.global_step)
                else:
                    if self.env_type == 'reacher_pose':
                        error = mean_final_dist + mean_final_ori_err
                    else:
                        error = mean_final_dist
                    if error < self.best_mean_err:
                        self.best_mean_err = error
                        is_best = True
                        print('*********************************************')
                        print('saving model with best mean error')
                        print('*********************************************')
                    else:
                        is_best = False
                    self.save_model(is_best=is_best, step=self.global_step)

    def train_net(self):
        batch_data = self.memory.sample(batch_size=self.batch_size)
        for key, value in batch_data.items():
            batch_data[key] = torch.from_numpy(value)
        obs0_t = batch_data['obs0']
        obs1_t = batch_data['obs1']
        obs0_t = self.normalize(obs0_t, self.obs_oms)
        obs1_t = self.normalize(obs1_t, self.obs_oms)
        obs0 = Variable(obs0_t).float().cuda()
        with torch.no_grad():
            vol_obs1 = Variable(obs1_t).float().cuda()

        rewards = Variable(batch_data['rewards']).float().cuda()
        actions = Variable(batch_data['actions']).float().cuda()
        terminals = Variable(batch_data['terminals1']).float().cuda()

        cri_q_val = self.critic(obs0, actions)
        with torch.no_grad():
            target_net_act = self.actor_target(vol_obs1)
            target_net_q_val = self.critic_target(vol_obs1, target_net_act)
            # target_net_q_val.volatile = False
            target_q_label = rewards
            target_q_label += self.gamma * target_net_q_val * (1 - terminals)
            target_q_label = target_q_label.detach()

        self.actor.zero_grad()
        self.critic.zero_grad()
        cri_loss = self.critic_loss(cri_q_val, target_q_label)
        cri_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(self.critic.parameters(),
                                          self.max_grad_norm)
        self.critic_optim.step()

        self.critic.zero_grad()
        self.actor.zero_grad()
        net_act = self.actor(obs0)
        net_q_val = self.critic(obs0, net_act)
        act_loss = -net_q_val.mean()
        act_loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm(self.actor.parameters(),
                                          self.max_grad_norm)
        self.actor_optim.step()

        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        return act_loss.cpu().data.numpy(), cri_loss.cpu().data.numpy()

    def normalize(self, x, stats):
        if stats is None:
            return x
        return (x - stats.mean) / stats.std

    def denormalize(self, x, stats):
        if stats is None:
            return x
        return x * stats.std + stats.mean

    def net_mode(self, train=True):
        if train:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()

    def load_model(self, step=None, load_dir=None):
        model_dir = self.model_dir
        if load_dir is not None:
            ckpt_file = os.path.join(self.load_dir, 'best.pth')
        else:
            if step is None:
                ckpt_file = os.path.join(model_dir, 'best.pth')
            else:
                ckpt_file = os.path.join(model_dir,
                                         '{:08d}.pth'.format(step))
        if not os.path.isfile(ckpt_file):
            raise ValueError("No checkpoint found at '{}'".format(ckpt_file))
        mutils.print_yellow('Loading checkpoint {}'.format(ckpt_file))
        checkpoint = torch.load(ckpt_file)
        if load_dir is not None:
            actor_dict = self.actor.state_dict()
            critic_dict = self.critic.state_dict()
            actor_pretrained_dict = {k: v for k, v in
                                     checkpoint['actor_state_dict'].items()
                                     if k in actor_dict}
            critic_pretrained_dict = {k: v for k, v in
                                      checkpoint['critic_state_dict'].items()
                                      if k in critic_dict}
            actor_dict.update(actor_pretrained_dict)
            critic_dict.update(critic_pretrained_dict)
            self.actor.load_state_dict(actor_dict)
            self.critic.load_state_dict(critic_dict)
            self.global_step = 0
        else:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.global_step = checkpoint['global_step']
        if step is None:
            mutils.print_yellow('Model ckpt step is: {}'
                                ''.format(checkpoint['ckpt_step']))

        self.warmup_iter += self.global_step
        mutils.print_yellow('Checkpoint loaded...')

    def save_model(self, is_best, step=None):
        if step is None:
            step = self.global_step
        ckpt_file = os.path.join(self.model_dir,
                                 '{:08d}.pth'.format(step))
        data_to_save = {'ckpt_step': step,
                        'global_step': self.global_step,
                        'actor_state_dict': self.actor.state_dict(),
                        'actor_optimizer': self.actor_optim.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'critic_optimizer': self.critic_optim.state_dict()}

        mutils.print_yellow('Saving checkpoint: %s' % ckpt_file)
        torch.save(data_to_save, ckpt_file)
        if is_best:
            torch.save(data_to_save, os.path.join(self.model_dir,
                                                  'best.pth'))

    def rollout(self, record=False):
        done_num = 0
        final_dist = []
        final_ori_err = []
        episode_length = []
        reward_list = []
        in_range_step_list = []
        for idx in range(self.test_case_num):
            obs = self.env.reset()
            total_reward = 0
            for t_rollout in range(self.rollout_steps):
                obs = obs[0].copy()
                act = self.policy(obs, stochastic=False).flatten()
                obs, r, done, info = self.env.step(act)
                total_reward += r
                if done:
                    done_num += 1
                    break
            dist = info['dist']
            final_dist.append(dist)
            if self.env_type == 'reacher_pose' or self.env_type == 'trajfollow_pose':
                ori_err = info['ori_err']
                final_ori_err.append(ori_err)
            if self.env_type == 'trajfollow' or self.env_type == 'trajfollow_pose':
                in_range_step = info['in_range_step']
                in_range_step_list.append(in_range_step)
            if record:
                if self.env_type == 'trajfollow':
                    print(f'Test case {idx}: dist({dist}), reward({total_reward}), in_range_step({in_range_step})')
                elif self.env_type == 'trajfollow_pose':
                    print(f'Test case {idx}: dist({dist}), ori_err({ori_err}), reward({total_reward}), in_range_step({in_range_step})')
                elif self.env_type == 'reacher_pose':
                    print(f'Test case {idx}: dist({dist}), ori_err({ori_err}), reward({total_reward})')
                elif self.env_type == 'reacher':
                    print(f'Test case {idx}: dist({dist}), reward({total_reward})')
            reward_list.append(total_reward)
            episode_length.append(t_rollout)
        final_dist = np.array(final_dist)
        reward_list = np.array(reward_list)
        final_ori_err = np.array(final_ori_err)
        in_range_step_list = np.array(in_range_step_list)
        mean_final_dist = np.mean(final_dist)
        mean_final_ori_err = np.mean(final_ori_err)
        mean_in_range_step = np.mean(in_range_step_list)
        mean_final_reward = np.mean(reward_list)
        succ_rate = done_num / float(self.test_case_num)
        if record:
            # with open(self.env_dir + '/test_data.json', 'w') as f:
            #     json.dump(final_dist.tolist(), f)
            if self.env_type == 'trajfollow' or self.env_type == 'trajfollow_pose':
                print('\nTest cases results:')
                print('------------In range step-------------')
                print("Minimum: {0:9.4f} Maximum: {1:9.4f}"
                    "".format(np.min(in_range_step_list), np.max(in_range_step_list)))
                print("Mean: {0:9.4f}".format(mean_in_range_step))
                print("Standard Deviation: {0:9.4f}".format(np.std(in_range_step_list)))
                print("Median: {0:9.4f}".format(np.median(in_range_step_list)))
                print("First quartile: {0:9.4f}"
                    "".format(np.percentile(in_range_step_list, 25)))
                print("Third quartile: {0:9.4f}"
                    "".format(np.percentile(in_range_step_list, 75)))
            print('\nTest cases results:')
            print('---------------Distance---------------')
            print("Minimum: {0:9.4f} Maximum: {1:9.4f}"
                "".format(np.min(final_dist), np.max(final_dist)))
            print("Mean: {0:9.4f}".format(mean_final_dist))
            print("Standard Deviation: {0:9.4f}".format(np.std(final_dist)))
            print("Median: {0:9.4f}".format(np.median(final_dist)))
            print("First quartile: {0:9.4f}"
                "".format(np.percentile(final_dist, 25)))
            print("Third quartile: {0:9.4f}"
                "".format(np.percentile(final_dist, 75)))
            if self.env_type == 'reacher_pose' or self.env_type == 'trajfollow_pose':
                print('---------------Orientation---------------')
                print("Minimum: {0:9.4f} Maximum: {1:9.4f}"
                    "".format(np.min(final_ori_err), np.max(final_ori_err)))
                print("Mean: {0:9.4f}".format(mean_final_ori_err))
                print("Standard Deviation: {0:9.4f}".format(np.std(final_ori_err)))
                print("Median: {0:9.4f}".format(np.median(final_ori_err)))
                print("First quartile: {0:9.4f}"
                    "".format(np.percentile(final_ori_err, 25)))
                print("Third quartile: {0:9.4f}"
                        "".format(np.percentile(final_ori_err, 75)))
            print('\n---------------Reward---------------')
            print("Minimum: {0:9.4f} Maximum: {1:9.4f}"
                  "".format(np.min(reward_list), np.max(reward_list)))
            print("Mean: {0:9.4f}".format(mean_final_reward))
            print("Standard Deviation: {0:9.4f}".format(np.std(reward_list)))
            print("Median: {0:9.4f}".format(np.median(reward_list)))
            print("First quartile: {0:9.4f}"
                  "".format(np.percentile(reward_list, 25)))
            print("Third quartile: {0:9.4f}"
                  "".format(np.percentile(reward_list, 75)))
            print('\n')
            print('Success rate:', succ_rate)
            print('\n')
        if self.env_type == 'trajfollow':
            return mean_final_dist, mean_final_reward, succ_rate, mean_in_range_step    
        elif self.env_type == 'trajfollow_pose':
            return mean_final_dist, mean_final_ori_err, mean_final_reward, succ_rate, mean_in_range_step
        elif self.env_type == 'reacher_pose':
            return mean_final_dist, mean_final_ori_err, mean_final_reward, succ_rate
        return mean_final_dist, mean_final_reward, succ_rate

    def log_model_weights(self):
        for name, param in self.actor.named_parameters():
            logger.logkv('actor/' + name,
                         param.clone().cpu().data.numpy())
        for name, param in self.actor_target.named_parameters():
            logger.logkv('actor_target/' + name,
                         param.clone().cpu().data.numpy())
        for name, param in self.critic.named_parameters():
            logger.logkv('critic/' + name,
                         param.clone().cpu().data.numpy())
        for name, param in self.critic_target.named_parameters():
            logger.logkv('critic_target/' + name,
                         param.clone().cpu().data.numpy())

    def random_action(self):
        act = np.random.uniform(-1, 1, self.ac_dim)
        return act

    def policy(self, obs, stochastic=True):
        self.actor.eval()
        ob = Variable(torch.from_numpy(obs)).float().cuda().view(1, -1)
        act = self.actor(ob)
        act = act.cpu().data.numpy()
        if stochastic:
            act = self.action_noise(act)
        self.actor.train()
        return act

    def cuda(self):
        self.critic.cuda()
        self.actor.cuda()
        if hasattr(self, 'critic_target'):
            self.critic_target.cuda()
            self.actor_target.cuda()
            self.critic_loss.cuda()

    def construct_optim(self, net, lr, weight_decay=None):
        if weight_decay is None:
            weight_decay = 0
        params = mutils.add_weight_decay([net],
                                         weight_decay=weight_decay)
        optimizer = optim.Adam(params,
                               lr=lr,
                               weight_decay=weight_decay)
        return optimizer

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(),
                                       source.parameters()):
            target_param.data.copy_(param.data)
