
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
#### 1. Install [Anaconda](https://www.anaconda.com/products/distribution#linux)
#### 2. Cloning the repo ###
`$ git clone git@github.com:zccccclin/ME5406P2.git`
#### 3. Conda environment setup ###
```bash
cd ME5406P2
conda env create -f environment.yml
conda activate proj2
```
If mpi4py fails to install, try `conda install mpi4py`.

### Training new models ###

### Testing trained models ###
### Testing existing models ###
Moving desired model from `Trained_models` into `./agent_ddpg/data/` folder and renamed it to corresponding environment name. Run the test commands as per previous section.

2. trajfollow
3. reacher_pose
4. trajfollow_pose
## parameters for tweaking
