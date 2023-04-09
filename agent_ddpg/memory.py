import numpy as np
from util.utilFunc import min2d

class RingBuffer(object):
    def __init__(self, maxlen, shape, dtype='float32'):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]

    def get_batch(self, idxs):
        return self.data[(self.start + idxs) % self.maxlen]

    def append(self, v):
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v

class ReplayMemory(object):
    def __init__(self, memory_capacity, obs_dim, act_dim, goal_dim):
        cap = int(memory_capacity)
        self.obs0 = RingBuffer(cap, shape=(int(obs_dim),))
        self.obs1 = RingBuffer(cap, shape=(int(obs_dim),))
        self.act = RingBuffer(cap, shape=(int(act_dim),))
        self.rew = RingBuffer(cap, shape=(1,))
        self.done = RingBuffer(cap, shape=(1,))
        self.achieved_goal = RingBuffer(cap, shape=(int(goal_dim),))

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
        self.obs0.append(obs0)
        self.obs1.append(obs1)
        self.act.append(act)
        self.rew.append(rew)
        self.done.append(done)
        self.achieved_goal.append(achieved_goal)