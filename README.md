# Sailboat Gridworld Trajectory Analysis using Hidden Markov Models
This project focuses on analyzing trajectories using Hidden Markov Models (HMMs) through three core components: **Likelihood Calculation**, **Decoding**, and **Learning**. Each component corresponds to specific stages of HMM-based analysis and can be run independently.

## Project Structure

- **Heatmaps/**: Contains heatmaps generated during trajectory analysis.
- **compare_trajectories/**: Includes scripts to compare original and decoded trajectories.
- **trajectories/**: Contains sample trajectory data files.
- **Baum Welch.py**: Implementation of the Baum-Welch algorithm for HMM parameter optimization.
- **Decoding.py**: Contains the Viterbi decoding algorithm for observation sequence decoding.
- **Likelihood.py**: Code to compute the likelihood of observed sequences given the model parameters.
- **Sampling.py**: Script for generating trajectory samples based on model parameters.
- **assignment 1.pdf**: The assignment guidelines for this project.
- **model_parameters.json**: JSON file containing initial model parameters, such as transition and emission probabilities.
- **notebook.ipynb**: Jupyter notebook with exploratory analysis, results, and visualizations.
- **report.pdf**: Detailed report of the project, methodology, and results.
- **trajectory_data.json**: Sample trajectory data used for HMM training and testing.
- **utils.py**: Utility functions used across different modules, including data preprocessing and visualization.


---

## 1. Likelihood

The **Likelihood** component calculates the probability (likelihood) of an observed sequence given the current HMM parameters. This is fundamental for understanding how well the HMM explains the observed data.

### Files
- **`Likelihood.py`**: Computes the forward probability of the observed trajectories to determine the likelihood.

### How to Run
To calculate the likelihood, execute the following command in the terminal:

```bash
python Likelihood.py
```

## Usage

This script reads from `trajectory_data.json` (the observed trajectory data) and `model_parameters.json` (the initial HMM parameters). It outputs the likelihood of the observed trajectory, providing a metric for model fit.

---

## 2. Decoding

The **Decoding** component uses the Viterbi algorithm to determine the most likely sequence of hidden states for a given observed trajectory. This helps to map observed data to inferred underlying states, giving insight into the structure of the trajectory.

### Files
- **`Decoding.py`**: Implements the Viterbi algorithm for state decoding based on observed trajectories.

### How to Run
To perform decoding using the Viterbi algorithm, execute:
```bash
python Decoding.py
```

## Usage

The script reads from `trajectory_data.json` (observed data) and `model_parameters.json` (HMM parameters). It outputs the most probable sequence of hidden states, either printed to the console or saved in a separate file. Decoding results can be visualized in the `compare_trajectories` folder, where decoded and original trajectories are compared.

---

## 3. Learning

The **Learning** component involves training the HMM parameters using the Baum-Welch algorithm, which iteratively adjusts the transition and emission probabilities to maximize the likelihood of the observed data. This is essential for creating a more accurate HMM.

### Files
- **`Baum Welch.py`**: Implements the Baum-Welch algorithm for HMM training.

### How to Run
To train the model parameters using the Baum-Welch algorithm, run:

```bash
python Baum Welch.py
```

## Usage

This script uses `trajectory_data.json` as input data and updates the parameters in `model_parameters.json`. The updated parameters can then be used in the likelihood and decoding steps to improve performance. Training outputs are saved for further analysis, and comparisons are stored in the `Heatmaps` folder for visualization.

---

## Additional Information

- **`Sampling.py`**: Generates synthetic trajectories based on the trained HMM. You can run this script with:
  
  ```bash
  python Sampling.py
  ```

### Visualizations

Generated heatmaps and trajectory comparisons can be found in the `Heatmaps` and `compare_trajectories` folders. These visualizations help illustrate the differences between the original and decoded trajectories.

This README provides an overview of each component, how they correspond to the project structure, and instructions on how to run each part. With these tools, you can explore HMMs and apply them to trajectory data analysis!
