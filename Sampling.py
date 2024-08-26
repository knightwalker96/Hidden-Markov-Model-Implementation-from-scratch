from utils import *
import random
import json
import seaborn as sns
import matplotlib.pyplot as plt

def generate_trajectories(starting_point):
    tuples_list = [starting_point]
    movements = [(1, 0), (0, 1), (0, -1), (-1, 0), (0, 0)]
    base_probabilities = [0.4, 0.3, 0.1, 0.1, 0.1]

    for _ in range(29):
        i, j = tuples_list[-1]
        probabilities = base_probabilities.copy()

        if i + 1 > 15:
            probabilities[4] += probabilities[0]
            probabilities[0] = 0
        if i - 1 < 1:
            probabilities[4] += probabilities[3]
            probabilities[3] = 0
        if j + 1 > 15:
            probabilities[4] += probabilities[1]
            probabilities[1] = 0
        if j - 1 < 1:
            probabilities[4] += probabilities[2]
            probabilities[2] = 0

        move = random.choices(movements, probabilities)[0]
        new_tuple = (i + move[0], j + move[1])
        tuples_list.append(new_tuple)

    return tuples_list

def sensor_probabilities(i, j):
    if i <= 9 and j <= 9:
        p1 = (18 - (i - 1) - (j - 1)) / 18
    else:
        p1 = 0
    if i <= 9 and j <= 15 and j >= 7:
        p2 = (18 - (i - 1) + (j - 15)) / 18
    else:
        p2 = 0
    if i <= 15 and i >= 7 and j <= 15 and j >= 7:
        p3 = (18 + (i - 15) + (j - 15)) / 18
    else:
        p3 = 0
    if i <= 15 and i >= 7 and j>=1 and j <= 9:
        p4 = (18 - (i - 15) - (j - 1)) / 18
    else:
        p4 = 0

    return [p1, p2, p3, p4]

def sensor(probabilities):
    tuple_elements = []

    for p in probabilities:
        element = random.choices([1, 0], weights=[p, 1-p])[0]
        tuple_elements.append(element)

    return tuple(tuple_elements)

if __name__ == "__main__":
    grid = Grid()

    starting_points = [(1, 1)]
    trajectories = []
    start_points = []
    end_points = []

    for point in starting_points:
        for _ in range(20):
            trajectory = generate_trajectories(point)
            trajectories.append(trajectory)
            start_points.append(trajectory[0])
            end_points.append(trajectory[-1])

    for idx, trajectory in enumerate(trajectories):
        print(f"Trajectory {idx+1}: {trajectory}")
        print(f"  Start Point: {start_points[idx]}")
        print(f"  End Point: {end_points[idx]}")
        print("-" * 50)

    for idx, trajectory in enumerate(trajectories):
        file_name = f"trajectory_{idx+1}.png"
        plot_trajectory(trajectory, save_path="trajectories", file_name=file_name)
    print("Trajectories have been plotted and saved.")

    trajectory_data = {
            "trajectories": [list(map(list, trajectory)) for trajectory in trajectories],  # Convert tuples to lists
            "senser_observations": []  # Will be filled below
        }

    senser_observations = []
    for trajectory in trajectories:
        temp_list = []
        for point in trajectory:
            i, j = point
            probabilities = sensor_probabilities(i, j)
            observation = sensor(probabilities)
            temp_list.append(observation)
        senser_observations.append(temp_list)
    for i in range(20):
        print(f"Senser observations for trajectory {i+1} are: {senser_observations[i]}")

    trajectory_data["senser_observations"] = senser_observations
    with open('trajectory_data.json', 'w') as f:
            json.dump(trajectory_data, f, indent=4)

    #Plotting the Sensor Detection Probabilites of each sensor in the grid
    sensor_grids = [np.zeros((15, 15)) for _ in range(4)]
    for i in range(1,16):
        for j in range(1,16):
            probabilities = sensor_probabilities(i, j)
            for k in range(4):
                sensor_grids[k][i-1, j-1] = probabilities[k]
    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    sns.set()
    for k in range(4):
        sns.heatmap(sensor_grids[k], ax=axes[k//2, k%2], cmap='Blues', annot=False, cbar=True,
                    square=True, xticklabels=False, yticklabels=False)
        axes[k//2, k%2].set_title(f'Sensor {k+1} Detection Probability')
    plt.tight_layout()
    plt.show()
