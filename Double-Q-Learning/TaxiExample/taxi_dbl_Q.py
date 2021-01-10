import gym
env = gym.make("Taxi-v3")
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

#Checking how many actions we have 
env.action_space

# Looking at the possible states. 
env.observation_space.n

print(env.step(1))
env.render()

# %%
#Example pulled from https://www.datahubbs.com/double-q-learning/


#Setting base Q learning and then double q learning for comparison. 
def q_learning(env, gamma=0.9, alpha=0.01, eps=0.05, num_episodes=1000):
    
    # Initialize Q-table
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = np.zeros(num_episodes)
    ep = 0
    while ep < num_episodes:
        s_0 = env.reset()
        done = False
        while done == False:
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                max_qs = np.where(np.max(Q[s_0])==Q[s_0])[0]
                action = np.random.choice(max_qs)

            s_1, r, done, _ = env.step(action)
            Q[s_0, action] += alpha*(r + gamma*np.max(Q[s_1]) - Q[s_0, action])
            rewards[ep] += r
            s_0 = s_1
            if done:
                ep += 1
    return rewards, Q

def double_q_learning(env, gamma=0.9, alpha=0.01, eps=0.05, num_episodes=1000):
    
    # Initialize Q-table
    Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = Q1.copy()
    rewards = np.zeros(num_episodes)
    ep = 0
    while ep < num_episodes:
        s_0 = env.reset()
        done = False
        while done == False:
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                q_sum = Q1[s_0] + Q2[s_0]
                max_qs = np.where(np.max(q_sum)==q_sum)[0]
                action = np.random.choice(max_qs)

            s_1, r, done, _ = env.step(action)
            # Flip a coin to update Q1 or Q2
            if np.random.rand() < 0.5:
                Q1[s_0, action] += alpha*(r + 
                    gamma*Q2[s_1, np.argmax(Q1[s_1])] - Q1[s_0, action])
            else:
                Q2[s_0, action] += alpha*(r + 
                    gamma*Q1[s_1, np.argmax(Q2[s_1])] - Q2[s_0, action])
            rewards[ep] += r
            s_0 = s_1
            if done:
                ep += 1
    return rewards, Q1, Q2

#%%
dq_rewards, Q1, Q2 = double_q_learning(env, num_episodes=10000, alpha=0.1)
q_rewards, Q = q_learning(env, num_episodes=10000, alpha=0.1)
window = 10
dq_avg_rewards = np.array([np.mean(dq_rewards[i-window:i])  
                           if i >= window
                           else np.mean(dq_rewards[:i])
                           for i in range(1, len(dq_rewards))
                          ])
q_avg_rewards = np.array([np.mean(q_rewards[i-window:i])  
                          if i >= window
                          else np.mean(q_rewards[:i])
                          for i in range(1, len(q_rewards))
                         ])

# %%

plt.figure(figsize=(12,8))
plt.plot(q_avg_rewards, label='Mean Q Rewards')
plt.plot(dq_avg_rewards, label='Mean Double-Q Rewards')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.show()

# %%
