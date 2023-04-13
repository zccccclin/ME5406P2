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
from traj_follow_env import TrajFollowEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--env', default='reacher', type=str)
    parser.add_argument('--moving_goal', action='store_true')
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--random_traj', action='store_true')
    parser.add_argument('--test_case_num', type=int, default=50)
    parser.add_argument('--num_iters', type=int, default=50000)
    parser.add_argument('--warmup_iter', type=int, default=50)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--rollout_steps', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--train_steps', type=int, default=100)
    parser.add_argument('--hid1_dim', type=int, default=128)
    parser.add_argument('--hid2_dim', type=int, default=256)
    parser.add_argument('--hid3_dim', type=int, default=256)
    parser.add_argument('--actor_lr', type=float, default=0.0001)
    parser.add_argument('--critic_lr', type=float, default=0.001)
    parser.add_argument('--critic_weight_decay', type=float, default=0.001)
    parser.add_argument('--use_her', action='store_true')
    parser.add_argument('--k_future', type=int, default=4)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--reward_scale', type=float, default=1)
    parser.add_argument('--ou_noise_std', type=float, default=0.2)
    parser.add_argument('--uniform_noise_high', type=float, default=0.5)
    parser.add_argument('--uniform_noise_low', type=float, default=-0.)
    parser.add_argument('--max_noise_dec_step', type=float, default=0.000)
    parser.add_argument('--tol', type=float, default=0.02)
    parser.add_argument('--random_prob', type=float, default=0.1)
    parser.add_argument('--normal_noise_std', type=float, default=0.1)
    parser.add_argument('--noise_type', default='uniform', choices=['uniform', 'ou_noise', 'gaussian'], type=str)
    parser.add_argument('--memory_limit', type=int, default=1e6)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--ob_norm', type=bool, default=False)
    parser.add_argument('--init_method', default='uniform', choices=['uniform', 'normal'], type=str)
    parser.add_argument('--max_grad_norm', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', '-rt', action='store_true')
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--resume_step', '-rs', type=int, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--gpu_id', default='0', type=str)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Logging setup
    np.set_printoptions(precision=4, suppress=True)
    save_dir = os.path.join(args.save_dir, args.env)
    log_dir = os.path.join(save_dir, 'logs')
    if not args.test and not args.resume:
        dele = input(f"Are you sure you want to override {args.env} model and log folders? (y/n)")
        if dele == 'y':
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
        os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'csv'])


    # Environment setup
    if args.env == 'trajfollow':
        env = TrajFollowEnv(render=args.render, random_traj=args.random_traj, train=not args.test, tolerance=0.05)
    else:
        env = ReacherEnv(render=args.render, moving_goal=args.moving_goal, random_start=args.random_start, train=not args.test, tolerance=args.tol, env_name=args.env)

    ddpg = DDPG(env=env, args=args)
    if args.test:
        ddpg.test()
    else:
        ddpg.train()

if __name__ == '__main__':
    main()

