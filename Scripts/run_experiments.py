import time
from typing import List, Tuple
from tqdm import tqdm
from torch import nn
import numpy as np
from Environment.cube import Cube
from MCTS.mcts import solve_cube
from utils.nnet_utils import get_device


def run_experiment(cube: Cube, model: nn.Module, num_cubes: int, scramble_length: int,
                   max_time: float, num_threads: int) -> Tuple[int, List[int]]:
    complete_solutions = 0
    solution_lengths = []

    for _ in tqdm(range(num_cubes), desc=f"Scramble length {scramble_length}"):
        initial_state, _ = cube.scramble_states([cube.solvedState], num_scrambles=scramble_length)
        solution, complete = solve_cube(cube, model, initial_state[0], max_time, num_threads)

        if complete:
            complete_solutions += 1
            solution_lengths.append(len(solution))

    return complete_solutions, solution_lengths


def main():
    cube = Cube(N=3)  # 3x3 Rubik's Cube
    model = cube.get_nnet_model()  # Create a new model instance
    # model_file = r"C:\Users\shrey\Desktop\RL rubicks cube\trained_models\3x3\maxback10\best_model_epoch_97.pt"
    model_file = r"C:\Users\shrey\Desktop\RL rubicks cube\trained_models\v2\3x3\best_model_epoch_9.pt"
    device = get_device()
    model = cube.load_nnet_model(model_file, model, device)

    num_cubes = 100
    scramble_lengths = [6, 7, 8, 9, 10]
    max_time = 30  # 10 seconds per cube
    num_threads = 4

    results = {}

    for scramble_length in scramble_lengths:
        start_time = time.time()
        complete_solutions, solution_lengths = run_experiment(cube, model, num_cubes, scramble_length, max_time, num_threads)
        total_time = time.time() - start_time

        results[scramble_length] = {
            "complete_solutions": complete_solutions,
            "average_solution_length": np.mean(solution_lengths) if solution_lengths else None,
            "min_solution_length": min(solution_lengths) if solution_lengths else None,
            "max_solution_length": max(solution_lengths) if solution_lengths else None,
            "total_time": total_time
        }

    # Print results
    print("\nExperiment Results:")
    for scramble_length, data in results.items():
        print(f"\nScramble Length: {scramble_length}")
        print(f"Complete Solutions: {data['complete_solutions']} / {num_cubes}")
        print(f"Solution Rate: {data['complete_solutions'] / num_cubes * 100:.2f}%")
        if data['average_solution_length']:
            print(f"Average Solution Length: {data['average_solution_length']:.2f}")
            print(f"Min Solution Length: {data['min_solution_length']}")
            print(f"Max Solution Length: {data['max_solution_length']}")
        print(f"Total Time: {data['total_time']:.2f} seconds")

    # Save results to file
    with open("experiment_results_v2.txt", "w") as f:
        for scramble_length, data in results.items():
            f.write(f"Scramble Length: {scramble_length}\n")
            f.write(f"Complete Solutions: {data['complete_solutions']} / {num_cubes}\n")
            f.write(f"Solution Rate: {data['complete_solutions'] / num_cubes * 100:.2f}%\n")
            if data['average_solution_length']:
                f.write(f"Average Solution Length: {data['average_solution_length']:.2f}\n")
                f.write(f"Min Solution Length: {data['min_solution_length']}\n")
                f.write(f"Max Solution Length: {data['max_solution_length']}\n")
            f.write(f"Total Time: {data['total_time']:.2f} seconds\n\n")


if __name__ == "__main__":
    main()
