import numpy as np
from util.utilFunc import min2d

class Buffer(object):
    def __init__(self, buffer_size, buffer_dim):
        self.max_size = int(buffer_size)
        self.start = 0
        self.length = 0
        self.data = np.zeros((int(buffer_size),) + (int(buffer_dim),)).astype(dtype='float32')

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise IndexError
        return self.data[(self.start + index) % self.max_size]
    
    def get_batch(self, batch_index):
        return self.data[(self.start + batch_index) % self.max_size]

    def add(self, data):
        if self.length < self.max_size:
            self.length += 1
        elif self.length == self.max_size:
            self.start = (self.start + 1) % self.max_size
        self.data[(self.start + self.length - 1) % self.max_size] = data

class ReplayMemory(object):
    def __init__(self, memory_capacity, obs_dim, act_dim, goal_dim):
        self.obs0 = Buffer(memory_capacity, obs_dim)
        self.obs1 = Buffer(memory_capacity, obs_dim)
        self.act = Buffer(memory_capacity, act_dim)
        self.rew = Buffer(memory_capacity, 1)
        self.done = Buffer(memory_capacity, 1)
        self.achieved_goal = Buffer(memory_capacity, goal_dim)

    def sample(self, batch_size):
        batch_index = np.random.random_integers(len(self.obs0) - 2, size=batch_size)
        obs0_batch = self.obs0.get_batch(batch_index)
        obs1_batch = self.obs1.get_batch(batch_index + 1)
        act_batch = self.act.get_batch(batch_index)
        rew_batch = self.rew.get_batch(batch_index)
        done_batch = self.done.get_batch(batch_index)
        achieved_goal_batch = self.achieved_goal.get_batch(batch_index)

        result = {
            "obs0": min2d(obs0_batch),
            "obs1": min2d(obs1_batch),
            "act": min2d(act_batch),
            "rew": min2d(rew_batch),
            "done": min2d(done_batch),
            "achieved_goal": min2d(achieved_goal_batch)
        }

        return result
    
    def append(self, obs0, act, rew, obs1, achieved_goal, done):
        self.obs0.add(obs0)
        self.obs1.add(obs1)
        self.act.add(act)
        self.rew.add(rew)
        self.done.add(done)
        self.achieved_goal.add(achieved_goal)