![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

# Multi-Armed Bandit Algorithms: UCB1 vs Epsilon-Greedy

This project implements and compares **Upper Confidence Bound (UCB1)** and **ε-Greedy** algorithms on the classic **multi-armed bandit** problem.

We evaluate and visualize the average reward over time for:

- **UCB1** (based on Chernoff-Hoeffding confidence bounds)
- **Fixed ε-Greedy** (ε = 0.1)
- **Decaying ε-Greedy** (ε = 1 / t)
