import numpy as np

class EpsilonGreedyDecayAgent:
    def __init__(self, k):
        self.k = k
        self.epsilon = 1
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.t = 0

    def select_action(self):
        self.t += 1
        self.epsilon = 1/(self.t + 1)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k)
        else:
            return np.argmax(self.values)
        
    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] += (reward - value) / n 
