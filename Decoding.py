import matplotlib.pyplot as plt
import json
from utils import *
from q1 import *
from Likelihood import *

def index_to_coordinate(index):
    i = (index // 15) + 1
    j = (index % 15) + 1
    return (i, j)

# Plotting the mean Manhattan Distance between the Original Trajectories and the Decoded Trajectories
def manhattan_distance(coord1, coord2):
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

def calculate_mean_manhattan_distance(true_trajectories, predicted_trajectories):
    num_trajectories = len(true_trajectories)
    trajectory_length = len(true_trajectories[0])

    mean_distances = np.zeros(trajectory_length)
    for t in range(trajectory_length):
        distances = []
        for i in range(num_trajectories):
            true_state = true_trajectories[i][t]
            predicted_state = predicted_trajectories[i][t]
            distance = manhattan_distance(true_state, predicted_state)
            distances.append(distance)
        mean_distances[t] = np.mean(distances)

    return mean_distances

def viterbi_algorithm(observations, transition_matrix, observation_matrix, final_state_probabilities, initial_state_probabilities):
    T = len(observations)
    obs_index = observation_to_index(observations[0])
    viterbi_probs = np.zeros((T, 225))
    backpointers = np.zeros((T, 225), dtype=int)
    viterbi_probs[0] = initial_state_probabilities * observation_matrix[obs_index]
    for t in range(1, T):
        obs_index = observation_to_index(observations[t])
        for j in range(225):
            max_prob, max_state = max(
                (viterbi_probs[t-1, i] * transition_matrix[i, j], i) for i in range(225)
            )
            viterbi_probs[t, j] = max_prob * observation_matrix[obs_index, j]
            backpointers[t, j] = max_state
    final_probs = viterbi_probs[T-1] * final_state_probabilities
    best_last_state = np.argmax(final_probs)
    best_path = np.zeros(T, dtype=int)
    best_path[-1] = best_last_state
    for t in range(T-2, -1, -1):
        best_path[t] = backpointers[t+1, best_path[t+1]]
    return best_path

def load_model_parameters(json_file="model_parameters.json"):
    with open(json_file, 'r') as f:
        data = json.load(f)
    transition_matrix = np.array(data['transition_matrix'])
    initial_state_probabilities = np.array(data['initial_state_probabilities'])
    final_state_probabilities = np.array(data['final_state_probabilities'])
    observation_probability_matrix = np.array(data['observation_probability_matrix'])

    return transition_matrix, initial_state_probabilities, final_state_probabilities, observation_probability_matrix

if __name__ == "__main__":
    with open('trajectory_data.json', 'r') as f:
        data = json.load(f)
    trajectories = data['trajectories']
    senser_observations = data['senser_observations']
    transition_probability_matrix, initial_state_probabilities, final_state_probabilities, observation_probability_matrix = load_model_parameters()

    decoded_trajectories = []
    for i, observations in enumerate(senser_observations):
        decoded_states = viterbi_algorithm(observations, transition_probability_matrix, observation_probability_matrix, final_state_probabilities, initial_state_probabilities)
        decoded_coordinates = [index_to_coordinate(state) for state in decoded_states]
        decoded_trajectories.append(decoded_coordinates)
    for i, decoded in enumerate(decoded_trajectories):
        print(f"Decoded trajectory {i+1}: {decoded}")

    for idx, (original_trajectory, decoded_trajectory) in enumerate(zip(trajectories, decoded_trajectories)):
        file_name = f"trajectory_{idx+1}_with_decoded.png"
        plot_trajectory_with_decoded(original_trajectory, decoded_trajectory, save_path="compare_trajectories", file_name=file_name)
    print("Original and decoded trajectories have been plotted and saved.")

    mean_manhattan_distances = calculate_mean_manhattan_distance(trajectories, decoded_trajectories)
    print("Mean Manhattan distances at each time step:", mean_manhattan_distances)
    plt.plot(range(len(mean_manhattan_distances)), mean_manhattan_distances, marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Mean Manhattan Distance')
    plt.title('Mean Manhattan Distance vs Time Step')
    plt.grid(True)
    plt.show()
