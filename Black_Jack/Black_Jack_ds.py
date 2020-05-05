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
        probs = [0.8, 0.2] if state [0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p = probs)
        next_state, reward, done, info = bj_env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
        return episode

def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    # intialize empty dictionaries of arrays
    returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda:np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # let us monitor our progress
        if i_episode % 1000 == 0::
            print("\rEpisode {}/{}".format((i_episode), num_episodes), end="")
