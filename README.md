# Robot arm reaching + task space trajectory following with DDPG + HER
This project explores the feasibility to solve robotic arm manipulation task using deep reinforcement learning method.

# Problem
Robotic arm reaching and task-space trajectory following tasks are complex problem that are currently solved by the use of inverse kinematics (IK). However, IK solution sometimes have long computational time and could potentially have singularity issues. 
As such, this project explores the possibility to accomplish these two tasks using DRL method.

# Folders:
- **environment** contains the custom pybullet environments built for reaching and trajectory following tasks
- **agent_ddpg** contains the modified DDPG agent used for this project. (See below for source)
- **agent_ppo** contains the PPO agent created following tutorial for this project.
- **Archive** contains the test environments and test IK solutions that was made when building the project codebase. Its outdated.
- **Trained_models** contains the trained models that could be used to replicate the results.

# Running the code
### Cloning the repo ###

### Conda environment setup ###

### Running existing models ###

### Training new models ###

### Testing trained models ###

# parameters for tweaking
