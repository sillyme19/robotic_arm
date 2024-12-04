import os
import gymnasium as gym
from stable_baselines3 import SAC
import panda_gym  # Ensure panda_gym is installed for "PandaPickAndPlace-v3"
from pybullet_utils import bullet_client
import pybullet
import time  # Import time module to control the delay between frames

# Initialize the environment with render_mode='human' for live visualization
client = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
env = gym.make("PandaPickAndPlace-v3", render_mode="human")

# Load the pretrained model from the `logs` folder
model = SAC.load("td3_withHer_1M", env=env)

# Set number of episodes to run
num_episodes = 10
frame_delay = 0.1  # Set delay time in seconds (adjust this to your preference)

# Run the model and visualize it for each episode
for episode in range(num_episodes):
    obs, _ = env.reset()  # Reset the environment at the beginning of each episode
    done = False

    print(f"Running episode {episode + 1}...")

    while not done:
        # Predict action from the model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take the action in the environment and observe the result
        obs, reward, done, truncated, _ = env.step(action)
        
        # The environment will automatically render to the screen due to `render_mode="human"`
        
        # Add a delay to slow down the simulation
        time.sleep(frame_delay)
    
    print(f"Episode {episode + 1} finished.")

# Close the environment after finishing
env.close()
client.disconnect()
