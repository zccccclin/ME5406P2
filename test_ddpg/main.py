import argparse
import json
import os
import random
import shutil
from typing import Pattern, SupportsBytes

import numpy as np
import torch

from ddpg import DDPG
from util import logger
import sys
sys.path.append('../environment')
from reacher_env import ReacherEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='random seed', type=int, default=1)
    parser.add_argument('--env', help='env to train', default='reacher', 
                        type=str)
    parser.add_argument('--num_iters', type=int, default=50000)
    parser.add_argument('--warmup_iter', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=200,
                        help='save model every n iterations')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='logging every n iterations')
    parser.add_argument('--rollout_steps', type=int, default=200,
                        help='number of steps in each episode')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--train_steps', type=int, default=100,
                        help='train steps per rollout')
    parser.add_argument('--hid1_dim', type=int, default=128,
                        help='first hidden layer size')
    parser.add_argument('--hid2_dim', type=int, default=256,
                        help='second hidden layer size')
    parser.add_argument('--hid3_dim', type=int, default=256,
                        help='third hidden layer size')
    parser.add_argument('--actor_lr', type=float, default=0.0001,
                        help='actor learning rate')
    parser.add_argument('--critic_lr', type=float, default=0.001,
                        help='critic learning rate')
    parser.add_argument('--critic_weight_decay', type=float,
                        default=0.001, help='critic weight_decay')
    parser.add_argument('--her', type=bool, default=False,
                        help='whether to use hindsight experience replay')
    parser.add_argument('--k_future', type=int, default=4,
                        help='hindsight experience replay k future goals')
    parser.add_argument('--tau', type=float, default=0.001,
                        help='tau for target network update')
    parser.add_argument('--reward_scale', type=float, default=1,
                        help='scaling reward')
    parser.add_argument('--ou_noise_std', type=float, default=0.2,
                        help='ou noise std')
    parser.add_argument('--uniform_noise_high', type=float, default=0.5,
                        help='uniform noise high limit')
    parser.add_argument('--uniform_noise_low', type=float, default=-0.,
                        help='uniform noise low limit')
    parser.add_argument('--max_noise_dec_step', type=float, default=0.000,
                        help='decreasing step of maximum noise level')
    parser.add_argument('--tol', type=float, default=0.04,
                        help='tolerance for distance to target')
    parser.add_argument('--random_prob', type=float, default=0.1,
                        help='probability of taking completely random action')
    parser.add_argument('--normal_noise_std', type=float, default=0.1,
                        help='std for normal noise')
    parser.add_argument('--noise_type', default='uniform',
                        choices=['uniform', 'ou_noise', 'gaussian'], type=str,
                        help='noise type for exploration')
    parser.add_argument('--memory_limit', type=int, default=1e6,
                        help='replay buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='gamma to calculate return')
    parser.add_argument('--ob_norm', type=bool, default=False,
                        help='normalize obsevations')
    parser.add_argument('--init_method', default='uniform',
                        choices=['uniform', 'normal'], type=str,
                        help='initialization method of params')
    parser.add_argument('--max_grad_norm', type=float, default=None,
                        help='maximum gradient norm in back-propagation')
    parser.add_argument('--robot_dir', type = str,
                        default='../assets/gen_gripper/3f_2j')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='ratio of training robots')
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--resume', '-rt', action='store_true',
                        help='resume training')
    parser.add_argument('--pretrain_dir', type=str, default=None)
    parser.add_argument('--resume_step', '-rs', type=int, default=None)
    parser.add_argument('--render', action='store_true',
                        help='render at test time')
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    np.set_printoptions(precision=4, suppress=True)
    log_dir = os.path.join(args.save_dir, 'logs')
    if not args.test and not args.resume:
        dele = input("Do you wanna recreate ckpt and log folders? (y/n)")
        if dele == 'y':
            if os.path.exists(args.save_dir):
                shutil.rmtree(args.save_dir)
        
        os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'csv'])
    env = ReacherEnv(render=args.render, moving_goal=True, train=not args.test, tolerance=0.02)
    print('environment made')
    ddpg = DDPG(env=env, args=args)
    if args.test:
        ddpg.test(render=args.render, slow_t=0.0)
    else:
        ddpg.train()

if __name__ == '__main__':
    main()

