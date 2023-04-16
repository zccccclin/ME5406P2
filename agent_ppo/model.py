import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, name, obs_dim, act_dim, hid1_dim, hid2_dim, hid3_dim):
        super(ActorCritic, self).__init__()
        self.name = name
        self.layer1 = nn.Linear(obs_dim, hid1_dim)
        self.layer2 = nn.Linear(hid1_dim, hid2_dim)
        self.layer3 = nn.Linear(hid2_dim, hid3_dim)
        self.outlayer = nn.Linear(hid3_dim, act_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        # activation function
        if self.name == "actor":
            out1 = F.selu(self.layer1(obs))
            out2 = F.selu(self.layer2(out1))
            out3 = F.selu(self.layer3(out2))
            out = torch.tanh(self.outlayer(out3))
            return out
        elif self.name == "critic":
            out1 = F.selu(self.layer1(obs))
            out2 = F.selu(self.layer2(out1)) 
            out3 = F.selu(self.layer3(out2))
            out = self.outlayer(out3)
            return out
    
