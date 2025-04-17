import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces
import time
class CropRowEnv(gym.Env):
    """
    Custom Gym environment for crop row path planning with two-component action:
      - The first element controls the robot’s orientation (0 for upward, 1 for downward).
      - The second element controls the movement command:
            * If < 2, it is a vertical move (0 = forward, 1 = backward).
            * If >= 2, it indicates a corridor-switching action.
    The observation is [corridor, vertical_position, orientation, driving_direction, sampling_x, sampling_y].
    A small extra penalty is given when the agent oscillates between forward and backward.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 num_crop_rows=5,      # Number of crop rows (vertical lines). Corridors = num_crop_rows - 1.
                 corridor_length=12,   # Vertical positions: -1 to 11.
                 max_episode_steps=100,
                ):
        super(CropRowEnv, self).__init__()
        self.max_episode_steps = max_episode_steps
        self.num_crop_rows = num_crop_rows
        self.num_corridors = num_crop_rows - 1  # corridors between crop rows
        self.corridor_length = corridor_length + 2  # add 2 as before

        # # Set the observation space:
        # # [corridor, vertical_position, orientation, driving_direction, sampling_x, sampling_y]
        self.observation_space = spaces.Box(
            low=np.array([0.5, -1.0, -1.0, 0.5, 0, 0], dtype=np.float32),
            high=np.array([
                self.num_corridors - 0.5,
                self.corridor_length - 1,
                1.0,  # orientation: 0 or 1
                self.num_corridors - 0.5,    # target corridor
                self.num_crop_rows - 1,
                self.corridor_length - 2
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Use a MultiDiscrete action space with 2 elements:
        # First element: orientation command, with 2 options (0 or 1)
        # Second element: movement command, with (2 + num_corridors) options
        self.action_space = spaces.MultiDiscrete([2, 2 + self.num_corridors])
        # self.action_space = spaces.MultiDiscrete([2, 1 + self.max_crop_rows])
        
        # These variables will be set by the agent:
        self.orientation = None      # fixed robot orientation (0: upward, 1: downward)
        self.drivingd = None         # vertical move command: 0 for forward, 1 for backward

        # Other environment variables:
        self.state = None
        self.sampling_point = None
        self.path = []  # store robot positions (corridor, pos)
        self.fig, self.ax = plt.subplots()
        # self.total_reward = 0
        self.turn = False

        # To help penalize oscillations:
        self.previous_action = None
        self.initial_pos = self.state
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        self.current_step = 0
        if seed is not None:
            np.random.seed(seed)

        # Randomize the start position
        self.state = (np.random.randint(self.num_corridors - 1) + 0.5,
                      np.random.randint(-1, self.corridor_length))
        # Randomize the goal position
        self.state = (1.5, 8)
        self.sampling_point = (np.random.randint(self.num_crop_rows),
                               np.random.randint(0, self.corridor_length - 1))
        self.sampling_point = (6,0)
        # For the new episode, leave orientation and driving direction undefined.
        self.orientation = np.random.choice([0, 1])
        self.orientation = 1
        self.drivingd = None

        self.path = [self.state]
        # self.total_reward = 0
        self.previous_action = None
        self.initial_pos = self.state
        # Return observation. If orientation or driving direction are not set, use -1.
        return np.array([
            self.state[0],
            self.state[1],
            self.orientation,      # orientation undefined
            self.state[0],        # driving direction undefined
            self.sampling_point[0],
            self.sampling_point[1]
        ], dtype=np.float32), {}

    def step(self, action):
        """Execute an action. The action is an array: [ori_action, move_action]."""
        self.current_step += 1
        corridor, pos = self.state
        self.current_corridor = corridor
        # Check goal condition.
        goal_crop, goal_pos = self.sampling_point
        # Unpack action: 
        # The first element specifies the robot's orientation (0: upward, 1: downward).
        # The second element specifies the movement command:
        #   <2: vertical move (0: forward, 1: backward)
        #   >=2: corridor switching
        move_action, ori_action = action
        self.orientation = ori_action
 
        done = False
        truncated = False

        # Check if at an end of the corridor.
        at_end = (pos == -1 or pos == self.corridor_length - 1)

        # Interpret the move_action:
        if move_action < 2:
            # Vertical movement.
            self.drivingd = move_action
        elif move_action >= 2:
            # Corridor switching command.
            value = move_action - 1.5  # target corridor index

        # Vertical movement (if move_action is 0 or 1):
        if move_action < 2:
            if self.drivingd == 0:  # forward
                if self.orientation == 0:  # robot facing up: forward means increasing pos
                    if pos < self.corridor_length - 1:
                        pos += 1
                elif self.orientation == 1:  # robot facing down: forward means decreasing pos
                    if pos > -1:
                        pos -= 1
            elif self.drivingd == 1:  # backward
                if self.orientation == 0:  # robot facing up: backward means moving down
                    if pos > -1:
                        pos -= 1
                elif self.orientation == 1:  # robot facing down: backward means moving up
                    if pos < self.corridor_length - 1:
                        pos += 1

        # Corridor switching: if move_action >= 2.
        elif move_action >= 2:
            if at_end:
                if 0.5 <= value < self.num_corridors:
                    corridor = value
                    self.turn = True
                    # When switching corridors, the agent will select new orientation in the next step.
                    # We set drivingd to None so the agent must choose a new movement command.
                    # self.orientation = None
                    self.drivingd = None

        # Update previous_action with the current action array.
        self.previous_action = action

        self.state = (corridor, pos)
        self.path.append(self.state)

        # Determine the crop row on the robot's left.
        # For a fixed orientation, if orientation==0 (up) then left_crop_row = corridor - 0.5; if 1 then corridor + 0.5.
        left_crop_row = None
        if self.orientation is not None:
            if self.orientation == 0:
                left_crop_row = corridor - 0.5
            elif self.orientation == 1:
                left_crop_row = corridor + 0.5

        if (pos == goal_pos) and (left_crop_row is not None) and (left_crop_row == goal_crop):
            done = True

        obs = np.array([
            corridor, pos,
            self.orientation if self.orientation is not None else -1.0,
            corridor,
            self.sampling_point[0], self.sampling_point[1]
        ], dtype=np.float32)
        if not done and self.current_step >= self.max_episode_steps:
            truncated = True

        return obs, done, truncated, {}

    def render(self, mode="human"):
        """Render the environment with the robot's full path."""
        self.ax.clear()
        # Draw crop rows as vertical lines.
        for i in range(self.num_crop_rows):
            self.ax.plot([i, i], [0.0, self.corridor_length - 2], color='green', linewidth=2)
        self.ax.set_xlim(-0.5, self.num_crop_rows)
        self.ax.set_ylim(-1.5, self.corridor_length - 0.5)
        self.ax.set_xlabel("Crop Rows")
        self.ax.set_ylabel("Position along corridor")
        self.ax.set_title("Crop Row Path Planning Environment")
        self.ax.set_aspect('equal')

        if len(self.path) > 1:
            xs, ys = zip(*self.path)
            self.ax.plot(xs, ys, '--', color='orange', linewidth=1, label="Path")

        robot_x, robot_y = self.state
        self.ax.plot(robot_x, robot_y, 'ro', markersize=12, label="Robot")

        # Draw orientation arrow (blue) based solely on the agent's chosen orientation.
        if self.orientation is not None:
            if self.orientation == 0:
                self.ax.arrow(robot_x, robot_y, -0.4, 0, head_width=0.2, head_length=0.2, fc='b', ec='b')
                self.ax.arrow(robot_x, robot_y, 0, 0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')
            elif self.orientation == 1:
                self.ax.arrow(robot_x, robot_y, 0.4, 0, head_width=0.2, head_length=0.2, fc='b', ec='b')
                self.ax.arrow(robot_x, robot_y, 0, -0.4, head_width=0.2, head_length=0.2, fc='b', ec='b')

        goal_x = self.sampling_point[0]
        goal_y = self.sampling_point[1]
        self.ax.plot(goal_x, goal_y, 'b*', markersize=15, label="Goal")

        self.ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.pause(0.3)

    def close(self):
        plt.close()

if __name__ == "__main__":
    
    num_crop_rows = 10
    corridor_length = 10
    env = CropRowEnv(num_crop_rows=num_crop_rows, corridor_length=corridor_length)
    state = env.reset()
    # print("Initial state:", state, "Orientation:", env.orientation, "Goal:", env.sampling_point)
    initial_corr, initial_pos = env.initial_pos
    goal_crop, goal_pos = env.sampling_point
    orientation = env.orientation
    done = False
    Overall_actions = []
    path1 = initial_pos + goal_pos
    path2 = (corridor_length - initial_pos) + (corridor_length - goal_pos)
    shortest_path = path1 if path1 <= path2 else path2
    start = time.time()
    # print("shortest path", shortest_path, path1)
    while not done:
        # For the first step, if drivingd is not set, choose a random vertical action for the second element.
        # print('new loop')
        if env.drivingd is None:
            # print("hereeee")
            if initial_corr == goal_crop - 0.5 and orientation == 1: 
                # print("here")
                if initial_pos < goal_pos:
                    move_action = 1
                elif initial_pos > goal_pos:
                    move_action = 0
            
            elif initial_corr == goal_crop + 0.5 and orientation == 0:
                # print("1")
                if initial_pos < goal_pos:
                    move_action = 0
                elif initial_pos > goal_pos:
                    move_action = 1     
            elif shortest_path == path1: # a little bit bug, the orientation is wrong when first entering the new row
                # print("first path 1", env.turn)
                if env.turn:
                    orientation = 1
                    move_action = 0
                else:
                    if orientation == 0:
                        move_action = 1
                    else:
                        move_action = 0
            else:
                # print("first path 2", env.turn)
                if env.turn:
                    orientation = 0
                    move_action = 0
                    # print("value", orientation, move_action)
                else:
                    if orientation == 0:
                        move_action = 0
                    else:
                        move_action = 1
                # print("value", orientation, move_action)
        else:
            if env.state[1] not in (-1, env.corridor_length - 1):
                # print("previous action is", env.previous_action)
                # print("here 1")
                move_action = env.previous_action[0] if env.previous_action is not None else np.random.choice([0, 1])
            elif not env.turn:
                # print("here 2")
                if initial_corr < goal_crop:
                    corridor = goal_crop - 0.5
                else:
                    corridor = goal_crop + 0.5
                move_action = int(2 + corridor)
            elif env.turn:
                # print("here 3")
                if shortest_path == path1:
                    # print("path1")
                    if initial_corr < goal_crop:
                        orientation = 1
                        move_action = 1
                    else:
                        orientation = 0
                        move_action = 0
                else:
                    # print("path2")
                    if initial_corr < goal_crop:
                        orientation = 1
                        move_action = 0
                    else:
                        orientation = 0
                        move_action = 1
        # Overall_actions.append([move_action, orientation])
        print(move_action, orientation)
        action = np.array([move_action, orientation])
        # action = [move_action, orientation]
        state, done, _, _ = env.step(action)
        # print(f"Action: {action} -> State: {state}, Orientation: {env.orientation}, Done: {done}")
        env.render()
        Overall_actions.append([int(move_action), int(orientation)])
    print("overall action sequence", Overall_actions)    
    distinct_actions = []
    for action in Overall_actions:
        if action not in distinct_actions:
            distinct_actions.append(action)

    # print("Distinct elements:", distinct_actions)
    env.close()
    end = time.time()
    print("total used time", end - start, distinct_actions)
    
