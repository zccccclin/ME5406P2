import argparse
import os
import random
import shutil
from typing import Pattern, SupportsBytes
import sys
sys.path.append('../environment')
import numpy as np
import torch

from ddpg import DDPG_Agent
from util import logger
from reacher_env import ReacherEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='random seed', type=int, default=1)
    parser.add_argument('--env', type=str, default='reacher')
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
    parser.add_argument('--use_her', type=bool, default=True)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--tolerance', type=float, default=0.02)
    parser.add_argument('--random_action_prob', type=float, default=0.1)
    parser.add_argument('--memory_capacity', type=int, default=1e6)
    parser.add_argument('--save_dir', type=str, default='./data')
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_iter', type=int, default=None)
    parser.add_argument('--moving_goal', type=bool, default=True)
    parser.add_argument('--test_case_num', type=int, default=50)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()

    # Print GPU info
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Logging setup
    log_location = os.path.join(args.save_dir, args.env)
    log_location = os.path.join(log_location, 'logs')
    if not args.test and not args.resume:
        check = input('Are you sure you want to overwrite the log directory? [y/n] ')
        if check == 'y':
            if os.path.exists(args.save_dir):
                shutil.rmtree(args.save_dir)
        os.makedirs(log_location, exist_ok=True)
    logger.configure(dir=log_location, format_strs=['tensorboard', 'csv'])

    # Environment setup
    if args.env == 'reacher':
        env = ReacherEnv(render=args.render, moving_goal=args.moving_goal, train=not args.test, tolerance=args.tolerance)

    # Agent setup
    agent = DDPG_Agent(env, args=args)
    if args.test:
        agent.test()
    else:
        agent.train()



                        


if __name__ == '__main__':
    main()
