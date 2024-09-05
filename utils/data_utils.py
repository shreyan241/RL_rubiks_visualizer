import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset


def load_dataset(file_path: str) -> TensorDataset:
    """
    Loads a dataset from a .pkl file and converts it to a PyTorch TensorDataset.

    Args:
        file_path (str): Path to the .pkl file.

    Returns:
        TensorDataset: The dataset loaded from the file, consisting of states, policy targets, and value targets.
    """
    # Open the pickle file and load the data
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Extract the states, policies, and values from the loaded data
    states = np.array(data['states'])  # Convert list of arrays to a single NumPy array
    policies = np.array(data['policies'])  # Assuming policy outputs are one-hot encoded
    values = np.array(data['rewards'])

    # Convert the data to PyTorch tensors
    states_tensor = torch.tensor(states, dtype=torch.float32)
    policies_tensor = torch.tensor(policies, dtype=torch.float32)
    values_tensor = torch.tensor(values, dtype=torch.float32)

    # Create a TensorDataset
    dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)

    return dataset
