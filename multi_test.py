import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
from stable_baselines3 import DQN
from main_dqn import CropRowEnv
import time
config = {
    "num_crop_rows": 65,
    "corridor_length": 10,
    "max_episode_steps": 50,
    "log_dir": "./logs",
    "test1_dir":"./models/curr_learning_51x10",
    "test2_dir":"./models/curr_learning_2",
}
with open("start_orient.txt", "r") as file:
    # Read all lines into a list
    start_orient = file.readlines()
with open("starting_points_world.txt", "r") as file:
    # Read all lines into a list
    start_point = file.readlines()   
with open("goals_world.txt", "r") as file:
    # Read all lines into a list
    goals = file.readlines() 
env = CropRowEnv(
        num_crop_rows=config["num_crop_rows"],
        corridor_length=config["corridor_length"],
        max_episode_steps=config["max_episode_steps"]
    )

num_crop_rows = 65
corridor_length = 207
def map_value_to_range(value, input_min=0, input_max=207, output_min=0, output_max=10):
    # Compute the step size
    step_size = (input_max - input_min) / (output_max - output_min)  # 20.7
    
    # Map the value to the new range and round to the nearest step
    mapped_value = round((value - input_min) / step_size)
    
    # Ensure the mapped value stays within the output range
    if value > input_max:
        return output_max  # Cap at 10 if the value exceeds 207
    return max(output_min, min(output_max, mapped_value))
def process_list(data):
    # Step 1: Remove the first element.
    if not data or len(data) < 2:
        return data
    data = data[1:]
    
    # Step 2: Remove consecutive duplicates.
    dedup = [data[0]]
    for item in data[1:]:
        if item != dedup[-1]:
            dedup.append(item)
    
    # Step 3: Process segments.
    result = []
    current_high = None  # Will store the last element (from the current segment) with first value > 2.
    # We define boundary elements as those whose first value is 0 or 1.
    for item in dedup:
        if item[0] in (0, 1):
            # Before adding a boundary, add the stored "high" element (if any) from the preceding segment.
            if current_high is not None:
                result.append(current_high)
                current_high = None  # reset for the next segment
            result.append(item)
        else:
            # For non-boundary items:
            # If the first value is greater than 2, update our current_high.
            if item[0] >= 2:
                current_high = item
            # If the first value is exactly 2, ignore it.
    return result
def calculate_distance(initial_corr, initial_pos, goal_pos, actions):
    distance = 0
    # print("actions", actions)
    for action in actions:
        if action == actions[0]:
            if action[0] == 0:
                if action[1] == 0:
                    distance += 207 - initial_pos
                else:
                    distance += initial_pos
            else:
                if action[1] == 0:
                    distance += initial_pos
                else:
                    distance += 207 - initial_pos
            # print("1 distance is", distance)
        elif action == actions[1]:
            distance += abs(action[0] - 1.5 - initial_corr) * 2.38
            # print("2 distance is", distance)
        elif action == actions[2]:
            if action[0] == 0:
                if action[1] == 0:
                    distance += goal_pos
                else:
                    distance += 207 - goal_pos
            else:
                if action[1] == 0:
                    distance += 207 -goal_pos
                else:
                    distance += goal_pos
            # print("3 distance is", distance)
    return distance

def test(num_episodes=10000):
    # Create environment once
    
    

    model = DQN.load(os.path.join(config["test2_dir"], "dqn_crop_65.zip"))
    
    success_count = 0
    steps_per_episode = []
    overall_actions = []
    total_distances = []
    success_time = []
    for episode in range(num_episodes):
        start_state = list(map(float, start_point[episode].strip().split()))
        initial_pos = start_state[1]
        start_state[1] = map_value_to_range(start_state[1])
        start_state = (int(start_state[0]/2.38) + 0.5, int(start_state[1]))
        
        sampling_point = list(map(float, goals[episode].strip().split()))
        goal_pos = sampling_point[1]
        sampling_point[1] = map_value_to_range(sampling_point[1])
        sampling_point = (int(sampling_point[0]/2.38), int(sampling_point[1]))
        # print(start_state, sampling_point)
        # print("sampling point", sampling_point[1])
        orientation = list(map(float, start_orient[episode].strip().split()))[0]
        orientation = 0 if orientation == 1 else 1
        # print("orientation", orientation)
        # state = env.reset(options={"start_state": start_state, "sampling_point": sampling_point, "orientation": orientation})

        obs, _ = env.reset(options={"start_state": start_state, "sampling_point": sampling_point, "orientation": orientation})
        done = False
        step_count = 0
        start = time.perf_counter()
        overall_action = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1
            # env.render()
            overall_action.append([int(action), int(env.orientation)]) 
            if terminated:
                end = time.perf_counter()
                success_time.append(end - start)
                success_count += 1
                overall_actions.append(overall_action)
                # steps_per_episode.append(step_count)
                # print("overall action", overall_action)
                filtered_actions = process_list(overall_action)
                # print(filtered_actions)
                total_distance = calculate_distance(start_state[0], initial_pos, goal_pos, filtered_actions)
                total_distances.append(total_distance)
            if truncated:
                success_time.append(0.0)
                total_distances.append(0.0)

                
    
    env.close()
    max_sequence = 0
    for actions in overall_actions:
        if len(actions) > max_sequence:
            max_sequence = len(actions)
            max_actions = actions
    print(f"Total Successful Episodes: {success_count}/{num_episodes}")
    # print(f"Average Steps per Episode: {np.mean(steps_per_episode):.2f}")
    # print(f"Min Steps: {np.min(steps_per_episode)}, Max Steps: {np.max(steps_per_episode)}")
    success_time = success_time[1:]
    print("average time for success episodes:", np.mean(success_time))
    # print("path legnth", total_distances)
    print("average path length", np.mean(total_distances), np.max(total_distances))
    # print("maximum actions", max_actions)
    with open("dqn_distances.txt", "w") as file:
        for item in total_distances:
            file.write(f"{item}\n")
    with open("dqn_times.txt", "w") as file:
        for item in success_time:
            file.write(f"{item}\n")
if __name__ == "__main__":
    test()