import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hid1_dim, hid2_dim, hid3_dim):
        super(Actor, self).__init__()
        self.h1 = nn.Linear(obs_dim, hid1_dim)
        self.h2 = nn.Linear(hid1_dim, hid2_dim)
        self.h3 = nn.Linear(hid2_dim, hid3_dim)
        self.out = nn.Linear(hid3_dim, action_dim)

        # initialize weights
        w1_size = self.h1.weight.data.size()[0]
        v = 1./np.sqrt(w1_size[0])
        self.h1.weight.data = torch.Tensor(w1_size).uniform_(-v, v)
        w2_size = self.h2.weight.data.size()[0]
        v = 1./np.sqrt(w2_size[0])
        self.h2.weight.data = torch.Tensor(w2_size).uniform_(-v, v)
        w3_size = self.h3.weight.data.size()[0]
        v = 1./np.sqrt(w3_size[0])
        self.h3.weight.data = torch.Tensor(w3_size).uniform_(-v, v)
        self.out.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state):
        h1_out = F.selu(self.h1(state))
        h2_out = F.selu(self.h2(h1_out))
        h3_out = F.selu(self.h3(h2_out))
        out = torch.tanh(self.out(h3_out))
        return out
            

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hid1_dim, hid2_dim, hid3_dim):
        super(Critic, self).__init__()
        self.h1 = nn.Linear(obs_dim, hid1_dim)
        self.h2 = nn.Linear(hid1_dim + action_dim, hid2_dim)
        self.h3 = nn.Linear(hid2_dim, hid3_dim)
        self.out = nn.Linear(hid3_dim, 1)

        # initialize weights
        w1_size = self.h1.weight.data.size()[0]
        v = 1./np.sqrt(w1_size[0])
        self.h1.weight.data = torch.Tensor(w1_size).uniform_(-v, v)
        w2_size = self.h2.weight.data.size()[0]
        v = 1./np.sqrt(w2_size[0])
        self.h2.weight.data = torch.Tensor(w2_size).uniform_(-v, v)
        w3_size = self.h3.weight.data.size()[0]
        v = 1./np.sqrt(w3_size[0])
        self.h3.weight.data = torch.Tensor(w3_size).uniform_(-v, v)
        self.out.weight.data.uniform_(-0.003, 0.003)

    def forward(self, state, action):
        h1_out = F.selu(self.h1(state))
        h2_out = F.selu(self.h2(torch.cat([h1_out, action], dim=1)))
        h3_out = F.selu(self.h3(h2_out))
        out = self.out(h3_out)
        return out
            
