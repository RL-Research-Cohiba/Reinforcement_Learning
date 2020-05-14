import main
import gym
gym.logger.set_level(40)  # Explicitly specify dtype as float32

env = gym.make('CartPole-v1')

env.seed(42)
obs = env.reset()

print(obs)
