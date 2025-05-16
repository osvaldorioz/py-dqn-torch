import gymnasium as gym
env = gym.make("CartPole-v1")
print(env.observation_space, env.action_space)