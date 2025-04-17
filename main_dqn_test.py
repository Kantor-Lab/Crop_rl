import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
class CropRowEnv(gym.Env):
    """
    Custom Gym environment for crop row path planning with orientation, sampling on crop rows,
    and path visualization.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 num_crop_rows=5,      # Number of crop rows (vertical lines). Corridors = num_crop_rows - 1.
                 corridor_length=12,    # Vertical positions: -1 to 11.
                 max_episode_steps=100,
                 max_crop_rows = 51
                ):
        super(CropRowEnv, self).__init__()

        self.max_episode_steps = max_episode_steps
        self.max_crop_rows = max_crop_rows
        self.num_crop_rows = num_crop_rows
        self.num_corridors = num_crop_rows - 1  # corridors between crop rows
        self.corridor_length = corridor_length +2

        # Randomize start position (corridor and vertical position)
        self.state = (np.random.randint(self.num_corridors-1) + 0.5,
                    np.random.randint(-1, self.corridor_length))
        # self.state = self.start_state

        # Randomize goal (sampling point)
        self.sampling_point = (np.random.randint(self.num_crop_rows),
                               np.random.randint(0, self.corridor_length - 1))
        
        #(corridor, vertical_position, orientation, target_corridor, sampling_x, sampling_y)
        self.observation_space = spaces.Box(
            low=np.array([0.5, -1.0, -1.0, 0.5, 0, 0], dtype=np.float32),
            high=np.array([
                self.max_crop_rows - 1.5,#self.num_corridors - 0.5, 
                self.corridor_length - 1, 
                1.0, #orientation
                self.max_crop_rows - 1.5, #self.num_corridors - 0.5, #target corridor
                self.max_crop_rows - 1,#self.num_crop_rows - 1, 
                self.corridor_length - 2
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Action space: MultiDiscrete with two components.
        # self.action_space = spaces.MultiDiscrete([3, self.num_corridors])
        # self.action_space = spaces.Discrete(3 * self.num_corridors)
        # self.action_space = spaces.Discrete(2 + self.num_corridors)
        self.action_space = spaces.Discrete(2 + self.max_crop_rows - 1)

        # Orientation: None until a vertical move is initiated.
        # 0: up (left is west; crop row = corridor)
        # 1: down (left is east; crop row = corridor + 1)
        self.orientation = None
        self.initial_orientation = None
        # Driving direction
        self.drivingd = None

        # Store the robot's path as a list of (x, y) positions.
        self.path = []

        # For visualization.
        self.fig, self.ax = plt.subplots()

        # Accumulated reward (total reward for the episode)
        self.total_reward = 0

        # Whether the robot turns
        self.turn = False

        # Previous step action
        self.previous_action = None

        #initial position
        self.initial_pos = self.state

    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        # Reset the random seed if provided
        self.current_step = 0

        if seed is not None:
            np.random.seed(seed)
        
        # Here you can randomize the state or set fixed values for testing
        self.state = (np.random.randint(self.num_corridors-1) + 0.5,
                    np.random.randint(-1, self.corridor_length))
        # self.state = (1.5, 6)
        # self.orientation = None
        self.orientation = np.random.choice([0, 1])
        self.initial_orientation = self.orientation
        # self.orientation = 0
        self.sampling_point = (np.random.randint(self.num_crop_rows),
                            np.random.randint(0, self.corridor_length - 1))
        # self.sampling_point = (6, 4)
        self.path = []
        # self.path.append(self._get_robot_coords())
        self.path.append(self.state)
        self.total_reward = 0
        self.drivingd = None
        self.previous_action = None
        self.initial_pos = self.state
        # Return the initial observation and an empty info dict (as required by Gymnasium)
        return np.array([self.state[0], self.state[1], -1.0, self.state[0], self.sampling_point[0], self.sampling_point[1]], dtype=np.float32), {}

    def _get_robot_coords(self):
        """Return the robot's current (x, y) coordinates for visualization."""
        corridor, pos = self.state
        # The x-coordinate is corridor index plus 0.5 (center of corridor).
        return (corridor + 0.5, pos)
    def check_corridor(self, reward, goal_crop):
        # if self.initial_pos[0] < goal_crop:
        #     if self.orientation == 1 and self.current_corridor == goal_crop - 0.5:
        #         reward += 1.0
        # elif self.initial_pos[0] > goal_crop:
        #     if self.orientation == 0 and self.current_corridor == goal_crop + 0.5:
        #         reward += 1.0
        if self.orientation == 1 and self.current_corridor == goal_crop - 0.5:
                reward += 1.0
        elif self.orientation == 0 and self.current_corridor == goal_crop + 0.5:
                reward += 1.0
        return reward
    def step(self, action):
        """Execute an action."""
        self.current_step += 1
        # Check goal condition.
        goal_crop, goal_pos = self.sampling_point
        corridor, pos = self.state
        self.current_corridor = corridor
        # Decode the action into action_type and value
        
        reward = -0.3  # Default step penalty
        done = False
        truncated = False  # Add truncated flag (not used in this environment)

        # Check if at an end of the corridor.
        at_end = (pos == -1 or pos == self.corridor_length - 1)
        # print("in step function, action type is", action_type)

        ###temporaly make sure robot not driving back and forth
        if not at_end and self.previous_action is not None:
            if (action == 0 and self.previous_action == 1) or (action == 1 and self.previous_action == 0):
                # If the new action is the opposite of the last action, override it
                # action = self.previous_action  
                # print("here")
                reward -= 0.3

        if action == 0:
            self.drivingd = 0
        elif action == 1:
            self.drivingd = 1
        elif 2 <= action <= self.num_crop_rows:
            value = action - 1.5 # target corridor index (0 to num_corridors-1)
        else:
            reward -= 0.5
        #action (0: forward; 1:backward) #orientation (0: upward; 1: downward)
        # print("action is", action)
        if 2 <= action <= self.num_crop_rows:
            # print("turning", action, at_end)
            if at_end:
                if 0.5 <= value < self.num_corridors:
                    if value != goal_crop - 0.5 and value != goal_crop + 0.5:
                        reward -= 0.5

                    corridor = value
                    self.orientation = np.random.choice([0, 1])
                    self.turn = True
                    reward = -0.3 * abs(self.current_corridor - corridor)
                    # if self.current_corridor < corridor:
                    #     if corridor == goal_crop - 0.5:
                    #         reward += 5.0
                    # elif self.current_corridor > corridor:
                    #     if corridor == goal_crop + 0.5:
                    #         reward += 5.0
                    # else:
                    #     if corridor == goal_crop -0.5 or corridor == goal_crop + 0.5:
                    #         reward += 5.0
                else:
                    reward = -1.5
            else:
                # print("here")
                reward = -1.6
        elif self.drivingd == 0: #forward 
            # print("forward", action, at_end)
            if action != 0:
                reward -= 1.0
            self.turn = False
            # reward = self.check_corridor(reward,goal_crop)
            if self.orientation == 1:
                if pos > -1:
                    pos -= 1
                else: 
                    reward = -1.5       
            elif self.orientation == 0:
                if pos < self.corridor_length - 1:
                    pos += 1
                else:
                    reward = -1.5
            else:
                reward = -1.6
        elif self.drivingd == 1: #backward
            # print("backward", action)
            # reward = self.check_corridor(reward,goal_crop)
            if action != 1:
                reward -= 1.0
            self.turn = False
            if self.orientation == 0:
                if pos > -1:
                    pos -= 1
                else:
                    reward = -1.5
            elif self.orientation == 1:
                if pos < self.corridor_length - 1:
                    pos += 1
                else:
                    reward = -1.5
            else:
                reward = -1.6
        else:
            # print("minus 0.6", action)
            reward = -1.6

        self.previous_action = action
        self.state = (corridor, pos)
        self.path.append(self.state)
        # self.path.append(self._get_robot_coords())

        # Determine the crop row on the robot's left.
        left_crop_row = None
        if self.orientation is not None:
            if self.orientation == 0:  # up: left is the crop row at index = corridor
                left_crop_row = corridor -0.5
            elif self.orientation == 1:  # down: left is the crop row at index = corridor + 1
                left_crop_row = corridor  + 0.5

        
        
        if (pos == goal_pos) and (left_crop_row is not None) and (left_crop_row == goal_crop):
            #switch to correct corridor
            if self.initial_pos[0] < goal_crop and self.orientation == 1:
                # if self.orientation == 1 and self.current_corridor == goal_crop - 0.5:
                reward += 20.0 
            elif self.initial_pos[0] > goal_crop and self.orientation == 0:
                # if self.orientation == 0 and self.current_corridor == goal_crop + 0.5:
                reward += 20.0

            reward += 40.0
            done = True
        # print("step reward", reward)
        # Accumulate the reward (for monitoring purposes)
        self.total_reward += reward
        # Return the observation, reward, terminated, truncated, and info
        orientation_value = self.orientation if self.orientation is not None else -1.0
        obs = np.array([corridor, pos, orientation_value, corridor, self.sampling_point[0], self.sampling_point[1]], dtype=np.float32)
        if not done and self.current_step >= self.max_episode_steps:
            truncated = True
        # if not done:  
        #     truncated = False  # Always allow the episode to continue
        if truncated:
            print("initial pos and goal", self.initial_pos, self.sampling_point)
        return obs, reward, done, truncated, {}
    

    def render(self, mode="human"):
        """Render the environment with the robot's full path."""
        self.ax.clear()
        # Draw crop rows as vertical lines from 0 to (num_crop_rows - 1)
        for i in range(self.num_crop_rows):
            self.ax.plot([i, i], [0.0, self.corridor_length - 2], color='green', linewidth=2)
        self.ax.set_xlim(-0.5, self.num_crop_rows)
        self.ax.set_ylim(-1.5, self.corridor_length - 0.5)
        self.ax.set_xlabel("Crop Rows")
        self.ax.set_ylabel("Position along corridor")
        self.ax.set_title("Crop Row Path Planning Environment")
        self.ax.set_aspect('equal')

        # Draw the robot's path.
        if len(self.path) > 1:
            xs, ys = zip(*self.path)
            self.ax.plot(xs, ys, '--', color='orange', linewidth=1, label="Path")  # Removed 'k--'

        # Draw the current robot position.
        # robot_x, robot_y = self._get_robot_coords()
        robot_x, robot_y = self.state
        self.ax.plot(robot_x, robot_y, 'ro', markersize=12, label="Robot")

        # Draw orientation arrows.
        if self.orientation is not None:
            if self.orientation == 0:
                self.ax.arrow(robot_x, robot_y, -0.4, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
                self.ax.arrow(robot_x, robot_y, 0, 0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')
            elif self.orientation == 1:
                self.ax.arrow(robot_x, robot_y, 0.4, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
                self.ax.arrow(robot_x, robot_y, 0, -0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')

        #Draw initial robot
        init_x, init_y = self.initial_pos
        self.ax.plot(init_x, init_y, 'ro', markersize=12, label="InitialRobot", alpha=0.7)

        # Draw orientation arrows.
        if self.initial_orientation is not None:
            if self.initial_orientation == 0:
                self.ax.arrow(init_x, init_y, -0.4, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
                self.ax.arrow(init_x, init_y, 0, 0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')
            elif self.initial_orientation == 1:
                self.ax.arrow(init_x, init_y, 0.4, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
                self.ax.arrow(init_x, init_y, 0, -0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')
        # Draw the sampling point (goal) on its crop row.
        goal_x = self.sampling_point[0]
        goal_y = self.sampling_point[1]
        self.ax.plot(goal_x, goal_y, 'b*', markersize=15, label="Goal")

        self.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.pause(0.3)
        plt.savefig("last_frame.png", bbox_inches='tight', pad_inches=0)

    def close(self):
        plt.close()

if __name__ == "__main__":
    num_crop_rows = 5
    corridor_length = 5
    env = CropRowEnv(num_crop_rows=num_crop_rows, corridor_length=corridor_length)
    state = env.reset()
    print("Initial state:", state, "Orientation:", env.orientation, "Goal:", env.sampling_point)
    print("Accumulated Reward: {:.2f}".format(env.total_reward))
    
    done = False
    while not done:
        # If orientation is not yet set, initialize with a vertical action
        if env.drivingd is None:
            random_choice = np.random.choice([0, 1])
            action = random_choice  # Randomly choose up (0) or down (1) to set orientation
        else:
            # If not at the boundary, keep moving vertically
            if env.state[1] not in (-1, env.corridor_length - 1):
                action = env.previous_action  # Move up
            else:
                # At boundary, switch corridor
                if env.turn:
                    action = np.random.choice([0, 1])  # Continue in the same orientation
                else:
                    possible_corridors = [i + 0.5 for i in range(env.num_corridors)]
                    target = np.random.choice(possible_corridors)
                    # possible_corridors = [c for c in range(env.num_corridors) if c != env.state[0]]
                    # target = np.random.choice(possible_corridors) if possible_corridors else env.state[0]
                    action = int(2 + target)  # Switch to a new corridor (action >= 2)
        
        state, reward, done, _, _ = env.step(action)
        print(f"Action: {action} -> State: {state}, Orientation: {env.orientation}, Step Reward: {reward:.2f}, Accumulated Reward: {env.total_reward:.2f}, Done: {done}")
        env.render()
        
    env.close()
    print("Episode finished with total reward: {:.2f}".format(env.total_reward))
