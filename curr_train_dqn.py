import os
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from main_dqn import CropRowEnv
import numpy as np

# Configuration parameters
config = {
    "curriculum_steps": 1000000,   # Train 1 million steps per stage
    "log_dir": "./logs",
    "test_dir": "./tests",
    "policy": "MlpPolicy",
    "learning_rate": 5e-5,
    "buffer_size": 200000,
    "learning_starts": 100000,
    "batch_size": 512,
    "gamma": 0.96,                # Lower gamma to make future rewards more discounted, encouraging shorter paths
    "target_update_interval": 8192,
    "render_freq": 1,
    "saved_model_path": "./logs/dqn_crop_51.zip"
}

# Define a function to create an environment based on the given number of crop rows
def create_env(num_rows, max_crop_rows):
    # Here, corridor_length is fixed at 6, and max_episode_steps can be adjusted as needed
    return CropRowEnv(num_crop_rows=num_rows, corridor_length=10, max_episode_steps=10000, max_crop_rows=max_crop_rows)

initial_num_rows = 10       # Initial number of crop rows       
num_intervals = 11          # Number of curriculum learning stages (how many stages you want)

# Calculate step_increment to ensure the number of intervals meets the requirement
step_increment = 5

final_num_rows = initial_num_rows + step_increment * num_intervals # Target number of crop rows
# Generate curriculum learning stages
curriculum_num_rows = list(range(initial_num_rows, final_num_rows + 1, step_increment))
print(curriculum_num_rows)
# curriculum_num_rows = [51]
final_num_rows = curriculum_num_rows[-1]
# initial_num_rows = 40
num = 1

model = None
initial_timesteps = 1000000

# model = DQN.load(config["saved_model_path"])

for num_rows in curriculum_num_rows:
    print(f"Training on environment with {num_rows} crop rows...")
    
    # Create the current stage's environment using make_vec_env
    env = make_vec_env(lambda: create_env(num_rows, final_num_rows), n_envs=1)
    # total_timesteps = (num_rows - initial_num_rows) // step_increment * initial_timesteps + initial_timesteps
    total_timesteps = num * initial_timesteps
    num += 1
    # num *= 1.5
    # If this is the first training stage, create a new model; otherwise, update the model's environment
    if model is None:
        model = DQN(
            config["policy"],
            env,
            policy_kwargs=dict(net_arch=[1024,1024]),  # Increase network capacity to handle more complex environments
            verbose=1,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            learning_starts=config["learning_starts"],
            batch_size=config["batch_size"],
            gamma=config["gamma"],
            target_update_interval=config["target_update_interval"],
            tensorboard_log=config["log_dir"],
            device='cuda' if torch.cuda.is_available() else 'cpu',
            exploration_initial_eps=1.0,
            exploration_final_eps=0.03,  # Extend exploration period
            exploration_fraction=0.5,
        )
    else:
        # Update the model's environment to continue training in the new stage
        model.set_env(env)
    
    # Train the current stage for the specified number of steps
    # model.learn(total_timesteps=config["curriculum_steps"])
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model after training in the current stage
    model.save(os.path.join(config["log_dir"], f"dqn_crop_{num_rows}"))
    # model.save(os.path.join(config["test_dir"], f"dqn_crop_{num_rows}"))
    env.close()

print("Curriculum learning completed!")

