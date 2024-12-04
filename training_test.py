from stable_baselines3 import TD3,SAC
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise
from pybullet_utils import bullet_client
import pybullet
import gymnasium as gym
import panda_gym  # Ensure panda_gym is installed for "PandaPickAndPlace-v3"
import numpy as np
import torch
import os
client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
# Set up the environment with the wrapper to ensure it returns five values
env = gym.make("PandaPickAndPlace-v3")
env = gym.wrappers.RecordEpisodeStatistics(env)  # Ensures compatibility with Stable Baselines3

# Custom reward normalization wrapper
class NormalizedRewardEnv(gym.Wrapper):
    def __init__(self, env):
        super(NormalizedRewardEnv, self).__init__(env)
        self.mean_reward = 0
        self.var_reward = 1

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # Normalize reward
        self.mean_reward = 0.99 * self.mean_reward + 0.01 * reward
        self.var_reward = 0.99 * self.var_reward + 0.01 * ((reward - self.mean_reward) ** 2)
        norm_reward = (reward - self.mean_reward) / (np.sqrt(self.var_reward) + 1e-8)
        return obs, reward, done, truncated, info

env = NormalizedRewardEnv(env)

# Define model class (TD3) and exploration noise
model_class = SAC
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))  # Increased noise for exploration

# Adjusted HER replay buffer parameters
replay_buffer_kwargs = dict(
    n_sampled_goal=20,  # Increased HER goals to sample per transition
    goal_selection_strategy="future",
    env=env,
)

# Updated model configuration
model = model_class(
    "MultiInputPolicy",
    env,
    action_noise=action_noise,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=replay_buffer_kwargs,
    verbose=1,
    buffer_size=1_000_000,
    learning_starts=10_000,
    batch_size=256,
    gamma=0.98,
    tau=0.005,
    learning_rate=3e-4,
    policy_kwargs=dict(net_arch=[512, 512, 256]),  # Deeper and wider architecture
    device="cuda" if torch.cuda.is_available() else "cpu",
)



# Train the model with extended timesteps
timesteps = 1500_000  # Increased training duration
model.learn(total_timesteps=timesteps)

# Save the model
model.save("td3_withHer_1M")

# Optionally, test the trained model
# obs, _ = env.reset()
# done = False
# while not done:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, _, _ = env.step(action)
#     env.render()
print(model.ep_success_buffer)
client.disconnect()