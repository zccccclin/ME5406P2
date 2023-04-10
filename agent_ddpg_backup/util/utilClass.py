import random
import numpy as np

class ActionNoise:
    def __init__(self, ll, ul):
        self.ll = ll
        self.ul = ul
        self.noise_level = random.uniform(self.ll, self.ul)

    def __call__(self, action):
        noise_act = np.random.normal(size=action.shape)
        action = action * (1 - self.noise_level) + noise_act * self.noise_level
        return action
    
    def reset(self):
        self.noise_level = random.uniform(self.ll, self.ul)
    
    def __repr__(self):
        return 'UniformNoise(low_limit={}, high_limit={})'.format(self.ll, self.ul)

