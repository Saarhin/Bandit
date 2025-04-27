import numpy as np

class UCB1Agent:

    def __init__(self, k, c):
        self.k = k
        self.c = c 
        self.counts = np.zeros(k)
        self.values = np.zeros(k)
        self.t = 0

    def select_action(self):
        self.t += 1
        ucb_values = np.zeros(self.k)
        for a in range(self.k):
            if self.counts[a] == 0:
                return a
            bonus = self.c * np.sqrt(np.log(self.t) / self.counts[a])
            ucb_values[a] = self.values[a] + bonus
        return np.argmax(ucb_values)
    
    def update(self, action, reward):
        self.counts[action] += 1
        n = self.counts[action]
        value = self.values[action]
        self.values[action] += (reward - value) / n