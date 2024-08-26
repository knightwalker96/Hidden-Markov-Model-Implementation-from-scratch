from utils import *
from q1 import *
from Likelihood import *
from Decoding import *

def calculate_entropy(original_matrix, pred_matrix):
    x = 0
    count = 0
    for i in range(len(original_matrix)):
        for j in range(len(original_matrix[0])):
            if original_matrix[i][j] > 0 and pred_matrix[i][j] > 0:
                x += original_matrix[i][j] * np.log(original_matrix[i][j] / pred_matrix[i][j])
                count += 1
    return x / count

def forward_algorithm_learning(observations, pi, B, T):
    alpha = np.zeros((20, 225))
    alpha[0, :] = pi * B[observation_to_index(observations[0]), :]
    for t in range(1, 20):
        for j in range(225):
            alpha[t, j] = np.sum(alpha[t - 1, :] * T[:, j]) * B[observation_to_index(observations[t]), j]
        alpha[t, :] /= np.sum(alpha[t, :])
    return alpha

def backward_algorithm_learning(observations, B, T):
    beta = np.zeros((20, 225))
    beta[20 - 1, :] = 1
    for t in range(20 - 2, -1, -1):
        for i in range(225):
            beta[t, i] = np.sum(T[i, :] * B[observation_to_index(observations[t + 1]), :] * beta[t + 1, :])
        beta[t, :] /= np.sum(beta[t, :])
    return beta

def baum_welch(all_observations, pi, B, T, original_T, original_B):
    kl_divergence_values_T = []
    kl_divergence_values_B = []
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}")
        xi_sum = np.zeros((225, 225))
        gamma_sum = np.zeros((20, 225))
        gamma_sum_all = np.zeros(225)
        observation_sum = np.zeros_like(B)
        for observations in all_observations:
            alpha = forward_algorithm_learning(observations, pi, B, T)
            beta = backward_algorithm_learning(observations, B, T)
            # E-step: Calculate xi and gamma
            for t in range(20 - 1):
                denom = np.sum(alpha[t, :] * beta[t, :])
                for i in range(225):
                    gamma_sum[t, i] += (alpha[t, i] * beta[t, i]) / denom
                    for j in range(225):
                        xi_sum[i, j] += (alpha[t, i] * T[i, j] * B[observation_to_index(observations[t + 1]), j] * beta[t + 1, j]) / denom
            gamma_sum[20 - 1, :] += alpha[20 - 1, :] / np.sum(alpha[20 - 1, :])
            for t in range(20):
                observation_idx = observation_to_index(observations[t])
                for j in range(225):
                    observation_sum[observation_idx, j] += gamma_sum[t, j]
                    gamma_sum_all[j] += gamma_sum[t, j]
        # M-step: Update transition and observation probability matrices
        new_T = np.zeros_like(T)
        for i in range(225):
            for j in range(225):
                new_T[i, j] = xi_sum[i, j] / np.sum(gamma_sum[:-1, i])
        new_T /= new_T.sum(axis=1, keepdims=True)
        new_B = observation_sum / gamma_sum_all[None, :]

        kl_divergence_T = calculate_entropy(original_T, new_T)
        kl_divergence_values_T.append(kl_divergence_T)
        kl_divergence_B = calculate_entropy(original_B, new_B)
        kl_divergence_values_B.append(kl_divergence_B)
        print(f"Iteration {iteration + 1}: KL Divergence of T = {kl_divergence_T}")
        print(f"Iteration {iteration + 1}: KL Divergence of B = {kl_divergence_B}")
        # Update T and B
        T = new_T
        B = new_B
    return T, B

if __name__ == "__main__":
    starting_points = [(1,1)]
    original_trajectories = []
    for point in starting_points:
        for _ in range(10000):
            trajectory = generate_trajectories(point)
            original_trajectories.append(trajectory)
    print(f"Total number of trajectories generated: {len(original_trajectories)}")


    all_observations = []
    for trajectory in original_trajectories:
        temp_list = []
        for point in trajectory:
            i, j = point
            probabilities = sensor_probabilities(i, j)
            observation = sensor(probabilities)
            temp_list.append(observation)
        all_observations.append(temp_list)
    print(f"Total number of observations: {len(all_observations)}")

    #all_observations = all_observations[:30]
    #original_trajectories = original_trajectories[:30]

    # Initialising the Transition probability matrix and the Observation probability matrix
    T = np.full((225,225), 1/225)
    B = np.full((16,225), 1/16)
    pi = np.full(225 , 1 / 225)
    original_T, _ , _ = generate_transition_matrix_from_trajectories(original_trajectories, grid_size=(15, 15))
    original_B = generate_observation_probability_matrix(all_observations, original_trajectories, grid_size = (15,15))
    num_iterations = 20
    updated_T, updated_B = baum_welch(all_observations, pi, B, T, original_T, original_B)
