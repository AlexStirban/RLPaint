from gym.envs.registration import register

register(
    id="telesketch/DiscreteTelesketch-v0",
    entry_point="telesketch.envs:DiscreteTelesketchEnv",
    max_episode_steps=300,
)