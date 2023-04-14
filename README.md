
# Manipulator Reaching & Task Space Trajectory Following with DDPG + HER
This project explores the feasibility to solve robotic arm manipulation task using deep reinforcement learning method.

## Problem
Robotic arm reaching and task-space trajectory following tasks are complex problem that are currently solved by the use of inverse kinematics (IK). However, IK solution sometimes have long computational time and could potentially have singularity issues. 
As such, this project explores the possibility to accomplish these two tasks using DRL method.

## Folders:
| Folder name      | Content |
| --------- | -----|
| environment  | Contains  custom pybullet environments built for reaching and trajectory following tasks. |
| agent_ddpg     |   contains the modified DDPG agent used for this project. ([Source](https://github.com/taochenshh/hcp)) |
| agent_ppo      |    contains the PPO agent created for this project. |
| Archive| contains the test environments and test IK solutions that was made when building the project codebase. (Outdated)|
|Trained_models|contains the trained models that could be used to replicate the results.|
## Running the code
The code is tested and working on Ubuntu 20.04 LTS
##### 1. Install [Anaconda](https://www.anaconda.com/products/distribution#linux)
##### 2. Cloning the repo ###
`git clone git@github.com:zccccclin/ME5406P2.git`
##### 3. Conda environment setup ###
```bash
cd ME5406P2
conda env create -f environment.yml
conda activate proj2
```
If mpi4py fails to install, try `conda install mpi4py`.

##### Training new models ###
There are total of 4 environments available for training, default training environment will be reacher:
1. reacher
2. trajfollow
3. reacher_pose
4. trajfollow_pose
To run training:
```bash
cd agent_ddpg
python main.py --env=reacher --use_her --moving_goal --num_iters=50000 --test_case_num=50 --reward_type=dense
```
Best model and log will be saved into `./data/[environment name]` by default. 
If training interrupted, it could be resumed from best model by adding `--resume` tag in the above terminal command.
##### Testing trained models ###
To run test cases on trained models:
`python main.py --env=reacher --test --render --moving_goal --random_start --test_case_num=50`
##### Testing existing models ###
Moving desired model from `Trained_models` into `./agent_ddpg/data/` folder and renamed it to corresponding environment name. Run the test commands as per previous section.

## parameters for tweaking
During training, parameters could be used to tweak training process:
##### Environment parameters
| Parameter name      | Type | Info |
| --------- | ----- | ----- |
| `--env`     |   str | environment name, 4 choices mentioned in Training new models subsection.|
| `--reward_type`      |    str | chocie of `--dense` or `--sparse` reward|
| `--moving_goal`| store_true| use this to enable random goal location for both training or testing.|
|`--random_start`|store_true| use this to let robot arm start at random position (only for reacher or reacher_pose).|
|`--random_traj`|store_true| use this to generate random trajectory for each iteration of training or testing.|
|`--dist_tol`|float|distance error tolerance value for reacher task.|
|`--ori_tol`|float| oritentation error tolerance value for reacher_pose task.|


##### Agent paramters 

| Parameter name  | Type | Info |
| --------- | ----- | ----- |
| `--test`  | store_true | run best model for test cases. |
| `--render`| store_true| use pybullet for rendering.|
| `--resume`  | store_true | resume from best model. |
| `--seed`  | int | random seed (Default is 1 to replicate my training). |
|`--test_case_num`|int| number of test case for testing model or model save evaluation during training.|
|`--num_iters`|int| number of total iterations for training.|
|`--warmup_iter`|int| number of warm up iterations for training.|
|`--save_interval`|int| model evaluation and saving interval for training.|
|`--log_interval`|int| Logging interval for training.|
|`--rollout_steps`|int| maximum steps for each rollout iteration.|
|`--batch_size`|int|batch size for training.|
|`--train_steps`|int|train steps for DDPG update.|
|`--hid1_dim`|int|hidden layer 1 size.|
|`--hid2_dim`|int|hidden layer 2 size.|
|`--hid3_dim`|int|hidden layer 3 size.|
|`--actor_lr`|float|Actor class learning rate.|
|`--critic_lr`|float|Critic class learning rate.|
|`--critic_weight_decay`|float|Critic class weight decay rate.|
|`--memory_limit`|int|limit for maximum memory storage.|
|`--use_her`|store_true|use this to enable hightsight experience relay for training.|
|`--k_future`|int|HER future iteration.|
|`--tau`|float|value of tau for training update.|
|`--gamma`|float|value of gamma for training update.|
|`--reward_scale`|float|value to scale the reward.|
|`--noise_type`|str|type of noise sampling method to be used.|
|`--ou_noise_std`|float|ou noise standard deviation value.|
|`--normal_noise_std`|float|normal noise standard deviation value.|
|`--uniform_noise_high`|float|max value of uniform noise sampling.|
|`--uniform_noise_low`|float|min value of uniform noise sampling.|
|`--max_noise_dec_step`|float|max uniform noise sample value decay step.|
|`--random_prob`|float|probability for random noise sampling.|
|`--ob_norm`|boolean|use this to enable or disable observation normalization.|




