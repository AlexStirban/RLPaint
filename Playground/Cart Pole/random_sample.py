import gym 

DEFAULT_ENV = "CartPole-v1"

# Default env
env = gym.make(DEFAULT_ENV)

# Run random action for the example
state = env.reset()

# Init a episode
while True:
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)
    env.render()

    if done:
        env.reset()

# Close environment
env.close()