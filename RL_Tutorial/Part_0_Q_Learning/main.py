import gym
import numpy as np

env = gym.make("FrozenLake-v0")

# Initialize table with all zeros
lr = .8
y = .95
num_episodes = 2000
# create lists to contain total rewards and steps per episode
#jList
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = Falsej = 0