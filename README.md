# deep-rl-autonomous-vehicle-safety
Deep Reinforcement Learning project for autonomous vehicle safety using DQN in the CARLA simulator.

📌 Overview

Autonomous driving is evolving rapidly with advances in AI and sensor systems, but **safety in dynamic environments** remains a major challenge.
Traditional supervised learning approaches require large labeled datasets and fail to generalize to **dangerous or rare safety-critical scenarios**.

This project implements **Deep Reinforcement Learning (DRL)** — specifically a **Deep Q-Network (DQN)** using **TensorFlow/Keras** — to train an autonomous driving agent inside the CARLA simulator.

The agent learns safe behavior (collision avoidance, speed control) by interacting with the environment and receiving reward signals.

## 🎯 Objectives


- Apply **Deep Reinforcement Learning (DRL)** — specifically the **DQN algorithm** — to improve autonomous vehicle decision-making.  
- Use the **CARLA simulator** to recreate complex and dynamic driving environments.  
- Develop a model capable of optimizing navigation while **reducing collision risks** across diverse scenarios.  
- Evaluate the agent’s performance and **compare it with existing state-of-the-art models**.
- Analyze limitations and propose future improvements  

## 1.  RL Overview
Unlike supervised or unsupervised learning, RL **does not require pre-collected data**. The agent generates data through interactions, learning by **trial and error** with **rewards** guiding its behavior.

**Key concepts:**
- **Agent:** Learns from environment interactions by observing states, taking actions, and receiving rewards.
- **Environment:** Where the agent acts and receives feedback.
- **State:** The current situation of the agent in the environment.
- **Reward:** Numerical feedback for each action, guiding learning.

## 2 Markov Decision Process
An RL problem can be modeled as an **MDP** `(S, A, P, R)`:
- **S:** set of states  
- **A:** set of actions  
- **P:** transition probabilities between states  
- **R:** reward function for actions in each state  

Goal: learn an **optimal policy π** that maximizes cumulative rewards over time.

## 3 Agent-Environment Interaction
At each time step `t`:
1. Agent in state `S_t`  
2. Takes action `A_t` → moves to `S_(t+1)`  
3. Receives reward `R_t`  

Objective: learn policy π that maximizes **expected cumulative reward**:  

\[
\pi^*(s) = \arg\max_\pi \mathbb{E}[\sum_{t=0}^\infty \gamma^t r_t | s_0 = s, \pi]
\]

