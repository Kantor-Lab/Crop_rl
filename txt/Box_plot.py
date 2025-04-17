import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    """
    Reads numerical data from a text file where each line contains one number.
    Returns the data as a NumPy array.
    """
    return np.loadtxt(filename)

def remove_outliers(data, threshold):
    """
    Removes any data points that are above or below the threshold.
    """
    return [x for x in data if x <= threshold]

# Read distance data from text files for each method
heuristic_distances = read_data("old_codes/search_distances.txt")
graph_distances     = read_data("old_codes/distances.txt")
dqn_distances       = read_data("old_codes/dqn_distances.txt")

# Read planning time data from text files for each method
heuristic_times = read_data("old_codes/search_times.txt")
graph_times     = read_data("old_codes/times.txt")
dqn_times       = read_data("old_codes/dqn_times.txt")


heuristic_times *= 1000
graph_times *= 1000
dqn_times *= 1000
dqn_times = remove_outliers(dqn_times, 10)
# Create subplots: one for distances and one for planning times
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Option 1: Adjust outlier properties using flierprops
flierprops = dict(marker='o', markersize=3, linestyle='none', alpha=0.5)

# Boxplot for distances with adjusted outliers
axes[0].boxplot([heuristic_distances, graph_distances, dqn_distances],
                labels=['Heuristic', 'Graph_A*', 'DQN'],
                flierprops=flierprops)
axes[0].set_title('Distances', fontsize=20)
axes[0].set_ylabel('Distance (m)', fontsize=20)
axes[0].set_xticklabels(['Heuristic', 'Graph_A*', 'DQN'], fontsize=20)

# Boxplot for planning times with adjusted outliers
axes[1].boxplot([heuristic_times, graph_times, dqn_times],
                labels=['Heuristic', 'Graph_A*', 'DQN'],
                flierprops=flierprops)
axes[1].set_title('Planning Times', fontsize=20)
axes[1].set_ylabel('Time (ms)', fontsize=20)
axes[1].tick_params(axis='x', labelsize=20)
axes[1].tick_params(axis='y', labelsize=20)
axes[1].set_xticklabels(['Heuristic', 'Graph_A*', 'DQN'], fontsize=20)

plt.tight_layout()
plt.show()
