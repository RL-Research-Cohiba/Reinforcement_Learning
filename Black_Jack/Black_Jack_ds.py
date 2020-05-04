import sys
import gym
import numpy as np
import random
from collections import defaultdict


bj_env = gym.make('Blackjack-v0')

def generate_episode_from_limit_stochastic(bj_env):
    episode = []
    state = bj_env.reset() # here, bj_enb is the env instance
    while True:
        probs = [0.8, 0.2] is state [0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p = probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
        return episode 