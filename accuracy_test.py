import torch
import numpy as np
from Environment.cube import Cube
from utils.nnet_utils import get_device

# Initialize the cube environment
cube = Cube(N=3)

# Initialize the model and load the weights
model_file = r"trained_models\3x3\best_model_epoch_97.pt"
device = get_device()
model = cube.get_nnet_model()
model = cube.load_nnet_model(model_file, model, device)


# Function to calculate accuracy
def calculate_accuracy(num_scrambles: int, num_cubes: int = 1000):
    # Generate solved states
    solved_states = [cube.solvedState for _ in range(num_cubes)]

    # Scramble the states using the specified number of scrambles
    scrambled_states, random_moves = cube.scramble_states(solved_states, num_scrambles=num_scrambles)

    correct_predictions = 0

    # Run prediction for each scrambled state
    for i, state in enumerate(scrambled_states):
        last_move_idx = random_moves[i][-1]  # Get the last scramble move applied

        with torch.no_grad():
            # Convert state to input and get policy from the model
            input_tensor = torch.FloatTensor(cube.state_to_nnet_input([state])[0])
            policy, _ = model(input_tensor)
            policy = policy.numpy().flatten()

            predicted_move = np.argmax(policy)  # Get the predicted move

        # Get the reverse of the last move applied
        reverse_move_idx = cube.moves_rev.index(cube.moves[last_move_idx])

        # Check if the prediction is correct
        if predicted_move == reverse_move_idx:
            correct_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / num_cubes
    return accuracy


# Test accuracy for different scramble levels
for num_scrambles in range(1, 6):  # Scramble levels from 1 to 5
    accuracy = calculate_accuracy(num_scrambles=num_scrambles, num_cubes=10000)
    print(f"Accuracy for {num_scrambles} scrambles: {accuracy * 100:.2f}%")
