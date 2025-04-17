# test_enhanced.py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from stable_baselines3 import DQN
from main_dqn_norm import CropRowEnv

def run_episode(env, model, num_episode, start_state, sampling_point, orientation):
    # Reset the environment with the fixed start state and sampling point
    obs, _ = env.reset(options={"start_state": start_state, "sampling_point": sampling_point, "orientation": orientation})
    
    done = False
    total_reward = 0
    episode_data = {
        'actions': [],
        'orientations': [],
        'reward': 0
    }
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # Record action and current orientation
        episode_data['actions'].append(int(action))
        episode_data['orientations'].append(int(env.orientation) if env.orientation is not None else -1)
        total_reward += reward
        
        # Add visualization: render the environment each step
        env.render()
        
        done = terminated or truncated
    
    episode_data['reward'] = total_reward
    print(f"Episode {num_episode} Reward: {total_reward:.2f}")
    return episode_data

def test():
    # Hardcoded parameters - modify these values as needed
    num_crop_rows = 6      # Number of crop rows
    corridor_length = 10   # Corridor length
    num_episodes = 10      # Number of episodes to run
    
    env = CropRowEnv(
        num_crop_rows=num_crop_rows,
        corridor_length=corridor_length,
        max_episode_steps=50
    )

    # Set fixed start state and sampling point
    start_state = (2.5, 11)    # Example: Start at corridor 2.5, vertical position 10
    sampling_point = (4, 6)   # Example: Goal at crop row 4, vertical position 12
    orientation = 1 # 0 is up, 1 is down
    model = DQN.load("./logs/dqn_crop_final.zip")
    best_episode = None
    
    for i in range(num_episodes):
        episode_data = run_episode(env, model, i+1, start_state, sampling_point, orientation)
        
        if best_episode is None or episode_data['reward'] > best_episode['reward']:
            best_episode = episode_data
    
    env.close()
    
    # Create combined array of (action, orientation)
    result_array = np.array([
        (a, o) for a, o in zip(best_episode['actions'], best_episode['orientations'])
    ], dtype=[('action', 'i4'), ('orientation', 'i4')])
    
    print("\nBest Episode Results:")
    print(f"Total Reward: {best_episode['reward']:.2f}")
    print("Action-Orientation Array:")
    print(result_array)
    result_array = result_array[1:]
    filtered_list = [result_array[0]]  # Start with the first element

    for i in range(1, len(result_array)):
        if result_array[i] != result_array[i - 1]:  # Compare full structured elements
            filtered_list.append(result_array[i])

    # Convert back to a structured NumPy array
    unique_arr = np.array(filtered_list, dtype=result_array.dtype)
    print(unique_arr)
    return result_array

if __name__ == "__main__":
    test()
