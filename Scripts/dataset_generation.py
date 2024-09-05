from typing import List
from Environment.cube import Cube, CubeState
from argparse import ArgumentParser
import pickle
import numpy as np
import os
import time
from multiprocessing import Queue, Process


def normalize_state(states: List[CubeState], N: int) -> List[np.ndarray]:
    """
    Normalize the states by dividing by (6 * N^2).

    Args:
        states (List[CubeState]): List of CubeState objects to be normalized.
        N (int): Size of the cube (e.g., 3 for a 3x3x3 cube).

    Returns:
        List[np.ndarray]: Normalized cube state representations.
    """
    # Stack the state colors into an array
    states_np = np.stack([state.colors for state in states], axis=0)

    # Normalize by dividing by (6 * N^2)
    normalized_states_np = states_np / (N**2)

    return normalized_states_np.astype(np.float32)


def generate_and_save_states(cube_length: int, num_states: int, back_max: int, filepath_queue: Queue):
    while True:
        filepath = filepath_queue.get()
        if filepath is None:
            break

        # Initialize cube
        cube = Cube(N=cube_length)

        # Generate data
        start_time = time.time()
        print("Generating data for %s" % filepath)
        states: List[CubeState]
        states, policies, rewards = cube.generate_states(num_states, (0, back_max))
        data_gen_time = time.time() - start_time

        # Normalize states
        normalized_states = normalize_state(states, cube_length)

        # Save data
        start_time = time.time()
        data = {
            'states': normalized_states,  # Save the colors attribute of CubeState
            'policies': policies,
            'rewards': rewards
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=-1)

        save_time = time.time() - start_time

        print("%s - Data Gen Time: %.2f s, Save Time: %.2f s" % (filepath, data_gen_time, save_time))


def main():
    parser = ArgumentParser()
    parser.add_argument('--cube_length', type=int, required=True,
                        help="Length of the cube's side (e.g., 3 for a 3x3x3 cube)")
    parser.add_argument('--back_max', type=int, required=True,
                        help="Maximum number of steps to take backwards from goal")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory to save files")
    parser.add_argument('--num_per_file', type=int, default=int(1e6), help="Number of states per file")
    parser.add_argument('--num_files', type=int, default=100, help="Number of files")
    parser.add_argument('--num_procs', type=int, default=1,
                        help="Number of processors to use when generating data")
    parser.add_argument('--start_idx', type=int, default=0, help="Start index for file name")

    args = parser.parse_args()

    # Ensure the data directory and specific cube length subdirectory exist
    cube_dir = os.path.join(args.data_dir, f"{args.cube_length}x{args.cube_length}")
    os.makedirs(cube_dir, exist_ok=True)

    # Make filepath queue
    filepath_queue = Queue()
    filepaths = [
        os.path.join(cube_dir, f"maxback{args.back_max}_idx{train_idx + args.start_idx}.pkl")
        for train_idx in range(args.num_files)
    ]
    for filepath in filepaths:
        filepath_queue.put(filepath)

    # Start data generation processes
    data_procs = []
    for _ in range(args.num_procs):
        data_proc = Process(target=generate_and_save_states,
                            args=(args.cube_length, args.num_per_file, args.back_max, filepath_queue))
        data_proc.daemon = True
        data_proc.start()
        data_procs.append(data_proc)

    # Stop data generation processes
    for _ in range(len(data_procs)):
        filepath_queue.put(None)

    for data_proc in data_procs:
        data_proc.join()


if __name__ == "__main__":
    main()
