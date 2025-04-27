import numpy as np
import matplotlib.pyplot as plt
from Multi_Armed.mab import MultiArmedBandit
from Multi_Armed.e_greedy import EpsilonGreedyAgent
from Multi_Armed.e_greedy_decay import EpsilonGreedyDecayAgent
from Multi_Armed.UCB1 import UCB1Agent

def run_experiments(agent_class, agent_kwargs, steps=10000, runs=2000):
    rewards = np.zeros((runs, steps))
    for run in range(runs):
        bandit = MultiArmedBandit(k=10)
        agent = agent_class(**agent_kwargs)
        for t in range(steps):
            action = agent.select_action()
            reward = bandit.pull(action)
            agent.update(action, reward)
            rewards[run, t] = reward

    mean_rewards = rewards.mean(axis=0)
    return mean_rewards

def main():
    e_agent_rewards = run_experiments(EpsilonGreedyAgent, {'k':10, 'epsilon': 0.1})
    e_decay_rewards = run_experiments(EpsilonGreedyDecayAgent, {'k':10})
    ucb_rewards = run_experiments(UCB1Agent, {'k':10, 'c':2})


    plt.figure(figsize=(12,8))
    plt.plot(ucb_rewards, label="UCB1 (c=2)")
    plt.plot(e_agent_rewards, label="ε-Greedy (ε=0.1)")
    plt.plot(e_decay_rewards, label="ε-Greedy (ε=1/t)")
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Performance Comparison of Bandit Algorithms')
    plt.legend()
    plt.grid()
    plt.show()







if __name__ == "__main__":
    main()