import numpy as np

class MultiArmedBandit:

    def __init__(self, k):
        self.k = k
        self.true_rewards = np.random.normal(0, 1, k)  # each arm ~ N(0,1)

    def pull(self, action):
        reward = np.random.normal(self.true_rewards[action], 1)  # Reward ~ N(mean, 1)
        return reward