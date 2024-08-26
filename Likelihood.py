from utils import *
from q1 import *
from collections import defaultdict
import json

def observation_to_index(observation):
    return observation[0] * 8 + observation[1] * 4 + observation[2] * 2 + observation[3]

def generate_transition_matrix_from_trajectories(trajectories, grid_size=(15, 15)):
    num_coords = grid_size[0] * grid_size[1]
    transition_matrix = np.zeros((num_coords, num_coords))
    initial_state_probabilities = np.zeros(num_coords)
    final_state_probabilities = np.zeros(num_coords)

    def index_from_coordinate(i, j, grid_size):
        return (i - 1) * grid_size[1] + (j - 1)

    transition_counts = defaultdict(lambda: defaultdict(int))
    state_counts = defaultdict(int)
    initial_state_counts = defaultdict(int)
    final_state_counts = defaultdict(int)
    total_trajectories = len(trajectories)

    for trajectory in trajectories:
        (i1, j1) = trajectory[0]
        (i2, j2) = trajectory[-1]
        start_index = index_from_coordinate(i1, j1, grid_size)
        end_index = index_from_coordinate(i2, j2, grid_size)
        initial_state_counts[start_index] += 1
        final_state_counts[end_index] += 1

        for k in range(len(trajectory) - 1):
            (i1, j1) = trajectory[k]
            (i2, j2) = trajectory[k + 1]
            current_index = index_from_coordinate(i1, j1, grid_size)
            next_index = index_from_coordinate(i2, j2, grid_size)
            transition_counts[current_index][next_index] += 1
            state_counts[current_index] += 1

    for current_index, next_states in transition_counts.items():
        total_transitions = state_counts[current_index]
        for next_index, count in next_states.items():
            transition_matrix[current_index][next_index] = count / total_transitions

    for state_index, count in initial_state_counts.items():
        initial_state_probabilities[state_index] = count / total_trajectories

    for state_index, count in final_state_counts.items():
        final_state_probabilities[state_index] = count / total_trajectories

    return transition_matrix, initial_state_probabilities, final_state_probabilities

def generate_observation_probability_matrix(senser_observations, trajectories, grid_size=(15, 15)):
    observation_probability_matrix = np.zeros((16, 225))
    observation_counts = defaultdict(lambda: defaultdict(int))
    state_counts = defaultdict(int)

    for trajectory, observations in zip(trajectories, senser_observations):
        for state, observation in zip(trajectory, observations):
            i, j = state
            state_index = (i - 1) * grid_size[1] + (j - 1)
            observation_index = observation_to_index(observation)
            observation_counts[state_index][observation_index] += 1
            state_counts[state_index] += 1

    # Calculate the probabilities
    for state_index in range(225):
        if state_counts[state_index] > 0:
            for observation_index in range(16):
                observation_probability_matrix[observation_index][state_index] = (
                    observation_counts[state_index][observation_index] / state_counts[state_index]
                )

    return observation_probability_matrix

def forward_algorithm(trajectory, senser_observations, transition_matrix, observation_matrix, initial_state_probabilities):
    T = len(trajectory)
    forward_probs = np.zeros((T, 225))
    initial_probs = initial_state_probabilities
    obs_index = observation_to_index(senser_observations[0])
    forward_probs[0] = initial_probs * observation_matrix[obs_index]
    for t in range(1, T):
        obs_index = observation_to_index(senser_observations[t])
        for j in range(225):
            forward_probs[t, j] = np.sum(forward_probs[t-1] * transition_matrix[:, j]) * observation_matrix[obs_index, j]
    observation_likelihood = np.sum(forward_probs[T-1, :])
    return observation_likelihood

def save_matrices_to_json(transition_matrix, initial_state_probabilities, final_state_probabilities, observation_probability_matrix, file_name="model_parameters.json"):
    data = {
        "transition_matrix": transition_matrix.tolist(),
        "initial_state_probabilities": initial_state_probabilities.tolist(),
        "final_state_probabilities": final_state_probabilities.tolist(),
        "observation_probability_matrix": observation_probability_matrix.tolist()
    }
    with open(file_name, 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    with open('trajectory_data.json', 'r') as f:
        data = json.load(f)
    trajectories = data['trajectories']
    senser_observations = data['senser_observations']
    transition_probability_matrix, initial_state_probabilities, final_state_probabilities = generate_transition_matrix_from_trajectories(trajectories, grid_size=(15, 15))
    observation_probability_matrix = generate_observation_probability_matrix(senser_observations, trajectories, grid_size=(15, 15))
    save_matrices_to_json(transition_probability_matrix, initial_state_probabilities, final_state_probabilities, observation_probability_matrix)
    observation_likelihoods = []
    for i in range(20):
        likelihood = forward_algorithm(trajectories[i], senser_observations[i], transition_probability_matrix, observation_probability_matrix, initial_state_probabilities)
        observation_likelihoods.append(likelihood)
    for i, likelihood in enumerate(observation_likelihoods):
        print(f"Observation Likelihood for trajectory {i+1}: {likelihood}")
