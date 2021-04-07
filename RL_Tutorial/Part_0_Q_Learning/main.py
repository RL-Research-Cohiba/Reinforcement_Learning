# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0

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
    d = False
    j = 0
    # The Q-Table Learning algorithm
    while j < 99:
        j += 1
        # Choose an action by greedily (with noise) picking form Q Table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_
                                               space.n*(1./(i+1))))
        