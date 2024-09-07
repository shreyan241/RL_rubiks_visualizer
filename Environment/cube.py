from typing import Dict, List, Tuple, Union
import numpy as np

from random import randrange
import torch
from torch import nn

from Environment.CubeConfig import adj_faces, faces_to_colors
from Models.nnet import ResnetModel


class CubeState:
    """
    Represents the state of a NxN Rubik's Cube using its color configuration.

    Attributes:
        colors (np.ndarray): A NumPy array representing the colors of the cube.
        hash (int): A cached hash value for the current state, used for quick comparisons.
    """
    __slots__ = ['colors', 'hash']

    def __init__(self, colors: np.ndarray) -> None:
        """
        Initializes the CubeState with the given color configuration.
        """
        self.colors: np.ndarray = colors
        self.hash = None

    def __hash__(self):
        """
        Returns a hash value for the current state. The hash is computed only once and cached.
        """
        if self.hash is None:
            self.hash = hash(self.colors.tostring())  # Compute and cache the hash
        return self.hash

    def __eq__(self, other):
        """
        Checks if this cube state is equal to another cube state.
        """
        return np.array_equal(self.colors, other.colors)


class Cube:
    """
    Rubibk's Cube Base Class
    """

    def __init__(self, N=3, current_state: CubeState = None):
        super().__init__()
        self.dtype = np.uint8
        self.N = N
        self.solvedState = self._get_solved_state()
        if not current_state:
            self.current_state = self._get_solved_state()
        else:
            self.current_state = current_state
        self.moves: List[Tuple[str, int]] = [(a, t) for a in ['U', 'D', 'L', 'R', 'B', 'F'] for t in [1, -1]]
        self.moves_rev: List[Tuple[str, int]] = [(a, t) for a in ['U', 'D', 'L', 'R', 'B', 'F'] for t in [-1, 1]]

        # get idxs changed for moves
        self.rotate_idxs_new: Dict[Tuple[str, int], np.ndarray] = dict()
        self.rotate_idxs_old: Dict[Tuple[str, int], np.ndarray] = dict()
        self.rotate_idxs_new, self.rotate_idxs_old = self._precompute_rotation_idxs(self.N, self.moves)

    def next_state(self, states: List[CubeState], move: int) -> Tuple[List[CubeState], List[float]]:
        states_np = np.stack([x.colors for x in states], axis=0)
        states_next_np = self._move_np_(states_np, move)

        states_next: List[CubeState] = [CubeState(x) for x in list(states_next_np)]

        return states_next

    def prev_state(self, states: List[CubeState], move: int) -> List[CubeState]:
        move: Tuple[str, int] = self.moves[move]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)

    def apply_move(self, move: int):
        """Applies a move to the current state of the cube."""
        next_states = self.next_state([self.current_state], move)
        self.current_state = next_states[0]  # Update the current state

    def get_current_state(self) -> CubeState:
        """Returns the current state of the cube."""
        return self.current_state

    def generate_solved_states(self, num_states: int, np_format: bool = False) -> Union[
                                                                                List[CubeState], np.ndarray]:
        if np_format:
            solved_np: np.ndarray = np.expand_dims(self.solvedState.colors.copy(), 0)
            solved_states: np.ndarray = np.repeat(solved_np, num_states, axis=0)
        else:
            solved_states: List[CubeState] = [CubeState(self.solvedState.colors.copy()) for _ in range(num_states)]

        return solved_states

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[CubeState],
                                                                                          np.ndarray,
                                                                                          np.ndarray]:
        """
        Generates states by scrambling the cubes from the solved state and calculates the corresponding
        final rewards and policies.

        Args:
            num_states (int): The number of initial solved states to generate.
            backwards_range (Tuple[int, int]): The range of scrambles to apply to each state.

        Returns:
            Tuple[List[CubeState], np.ndarray, np.ndarray]:
                - A list of CubeState objects representing the final scrambled states.
                - A single policy array for each final scrambled state.
                - A single reward array for each final scrambled state.
        """
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states_np: np.ndarray = self.generate_solved_states(num_states, np_format=True)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        final_policies = np.zeros((num_states, num_env_moves))
        final_rewards = np.zeros(num_states)

        # Process states with 0 scrambles
        zero_scramble_idxs = np.where(scramble_nums == 0)[0]
        if len(zero_scramble_idxs) > 0:
            final_rewards[zero_scramble_idxs] = 1.0
            # Policies remain as zeros (no move needed)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_np[idxs], rewards = self._move_np_scramble(states_np[idxs], move, scramble_nums[idxs])

            # Generate policy vectors (one-hot encoded for the reverse move)
            reverse_move_idx = self.moves_rev.index(self.moves[move])
            final_policies[idxs, :] = 0  # Reset the policy array
            final_policies[idxs, reverse_move_idx] = 1

            # Store the final rewards
            final_rewards[idxs] = rewards

            num_back_moves[idxs] += 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states = [CubeState(x) for x in list(states_np)]

        return states, final_policies, final_rewards

    def is_solved(self, states: List[CubeState]) -> np.ndarray:
        states_np = np.stack([state.colors for state in states], axis=0)
        is_equal = np.equal(states_np, np.expand_dims(self.solvedState.colors, 0))

        return np.all(is_equal, axis=1)

    def get_num_moves(self) -> int:
        return len(self.moves)

    def state_to_nnet_input(self, states: List[CubeState]) -> List[np.ndarray]:
        """ Converts a list of CubeStates into a neural network input representation. """
        states_np = np.stack([state.colors for state in states], axis=0)

        representation_np: np.ndarray = states_np / (self.N ** 2)
        representation_np: np.ndarray = representation_np.astype(self.dtype)

        representation: List[np.ndarray] = [representation_np]

        return representation

    def get_nnet_model(self) -> nn.Module:
        """ Initializes and returns a neural network model. """
        state_dim: int = (self.N ** 2) * 6
        nnet = ResnetModel(state_dim=state_dim,
                           one_hot_depth=6,
                           h1_dim=5000,
                           resnet_dim=1000,
                           num_resnet_blocks=4,
                           policy_out_dim=12,
                           value_out_dim=1,
                           batch_norm=True,
                           dropout=0.1)
        return nnet

    def load_nnet_model(self, model_file: str, nnet: nn.Module, device=None):
        """
        Loads the model state dict from the checkpoint file.

        Args:
            model_file (str): The path to the checkpoint file.
            nnet (nn.Module): The model to load the state dict into.
            device (torch.device, optional): The device to load the model onto.

        Returns:
            nn.Module: The model with loaded weights.
        """
        if device is None:
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=device, weights_only=True)

        # Load model state
        nnet.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {model_file}")

        # Set the model to evaluation mode
        nnet.eval()

        return nnet

    def expand(self, states: List[CubeState]) -> List[List[CubeState]]:
        """
        Expands the current list of CubeStates by applying all possible moves to each state.

        This function simulates every possible move from each of the provided states, resulting in a list of
        new states for each original state.

        Args:
            states (List[CubeState]): A list of CubeState objects representing the current states of the cube.

        Returns:
            List[List[CubeState]]:
                A list of lists where each sublist contains the resulting CubeState objects after applying
                all possible moves to the corresponding input state.
        """
        # Initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_np: np.ndarray = np.stack([state.colors for state in states], axis=0)

        # Initialize container for expanded states
        states_exp = np.empty((num_states, num_env_moves, states_np.shape[1]), dtype=states_np.dtype)

        # Apply each possible move to all states
        for move_idx in range(num_env_moves):
            states_next_np = self._move_np_(states_np, move_idx)
            states_exp[:, move_idx, :] = states_next_np

        # Convert expanded states to list of lists of CubeState objects
        states_exp_list: List[List[CubeState]] = [
            [CubeState(states_exp[i, j]) for j in range(num_env_moves)] for i in range(num_states)
        ]

        return states_exp_list

    def _move_np_(self, states_np: np.ndarray, move: int):
        action_key = self.moves[move]
        states_next_np = states_np.copy()
        states_next_np[:, self.rotate_idxs_new[action_key]] = states_np[:, self.rotate_idxs_old[action_key]]

        return states_next_np

    def _move_np_scramble(self, states_np: np.ndarray, move: int, scramble_nums: np.ndarray):
        """
        Applies a move to the given states and computes the resulting states and rewards.

        Args:
            states_np (np.ndarray): A NumPy array of states to be transformed.
            move (int): The move to apply.
            scramble_nums (np.ndarray): A NumPy array of current scramble numbers.

        Returns:
            np.ndarray: The resulting states after the move.
            np.ndarray: The rewards associated with the move.
        """
        action_key = self.moves[move]
        states_next_np = states_np.copy()
        states_next_np[:, self.rotate_idxs_new[action_key]] = states_np[:, self.rotate_idxs_old[action_key]]

        # Calculate rewards based on the current scramble number
        rewards = 1 / (scramble_nums + 1)

        return states_next_np, rewards

    def scramble_states(self, states: List[CubeState], num_scrambles: int) -> List[CubeState]:
        """
        Scrambles a batch of CubeState objects by applying a sequence of random moves to each cube.

        Args:
            states (List[CubeState]): A list of CubeState objects representing the cubes to be scrambled.
            num_scrambles (int): The number of random scrambles to apply to each CubeState.

        Returns:
            List[CubeState]: A list of CubeStates after scrambling.
        """
        # Generate random moves for each cube in the batch
        # Shape: (len(states), num_scrambles) -> A sequence of moves for each cube
        random_moves = np.random.choice(len(self.moves), size=(len(states), num_scrambles))

        # Apply the sequence of random moves to each cube in the batch
        scrambled_states = self.apply_move_sequence(states, random_moves)

        return scrambled_states, random_moves

    def apply_move_sequence(self, states: List[CubeState], move_sequences: np.ndarray) -> List[CubeState]:
        """
        Applies a sequence of moves to each CubeState object in the batch using the _move_np_batch function.

        Args:
            states (List[CubeState]): A list of CubeState objects to apply the move sequences to.
            move_sequences (np.ndarray): A 2D array of move indices where each row is a sequence of moves for the
                                        corresponding CubeState.

        Returns:
            List[CubeState]: The CubeState objects after applying the move sequences.
        """
        # Convert CubeState objects to a NumPy array of colors
        states_np = np.stack([x.colors for x in states], axis=0)

        # Apply the batch of move sequences to the batch of states
        for i in range(move_sequences.shape[1]):  # Loop over number of scrambles
            states_np = self._move_np_batch(states_np, move_sequences[:, i])

        # Convert the updated states back to CubeState objects
        states_next = [CubeState(colors) for colors in states_np]
        return states_next

    def _move_np_batch(self, states_np: np.ndarray, moves: np.ndarray) -> np.ndarray:
        """
        Applies a batch of moves to a batch of cube states.

        Args:
            states_np (np.ndarray): A NumPy array of cube states (batch_size, cube_size).
            moves (np.ndarray): A NumPy array of move indices, one for each state in the batch.

        Returns:
            np.ndarray: Updated states after applying the moves.
        """
        # Copy the states to avoid modifying the original states
        states_next_np = states_np.copy()

        # Vectorized move application: apply each move to the corresponding state
        for i, move in enumerate(moves):  # For each state, apply the corresponding move
            action_key = self.moves[move]
            states_next_np[i, self.rotate_idxs_new[action_key]] = states_np[i, self.rotate_idxs_old[action_key]]

        return states_next_np

    def _precompute_rotation_idxs(self, cube_size: int,
                                  moves: List[Tuple[str, int]]) -> Tuple[
                                      Dict[Tuple[str, int], np.ndarray],
                                      Dict[Tuple[str, int], np.ndarray]]:
        """Precompute the rotation indices for each possible move on the Rubik's Cube."""
        rotate_idxs_new: Dict[str, np.ndarray] = dict()
        rotate_idxs_old: Dict[str, np.ndarray] = dict()
        for move in moves:
            f: str = move[0]
            sign: int = move[1]

            rotate_idxs_new[move] = np.array([], dtype=int)
            rotate_idxs_old[move] = np.array([], dtype=int)

            colors = np.zeros((6, cube_size, cube_size), dtype=np.int64)
            colors_new = np.copy(colors)

            # WHITE:0, YELLOW:1, BLUE:2, GREEN:3, ORANGE: 4, RED: 5
            adj_idxs = {
                        # Upper face (U) -> adjacent faces: L, R, B, F
                        0: {
                            2: [range(0, cube_size), cube_size - 1],  # Left face (L)
                            3: [range(0, cube_size), cube_size - 1],  # Right face (R)
                            4: [range(0, cube_size), cube_size - 1],  # Back face (B)
                            5: [range(0, cube_size), cube_size - 1],  # Front face (F)
                        },
                        # Down face (D) -> adjacent faces: L, R, B, F
                        1: {
                            2: [range(0, cube_size), 0],              # Left face (L)
                            3: [range(0, cube_size), 0],              # Right face (R)
                            4: [range(0, cube_size), 0],              # Back face (B)
                            5: [range(0, cube_size), 0],              # Front face (F)
                        },
                        # Left face (L) -> adjacent faces: U, D, B, F
                        2: {
                            0: [0, range(0, cube_size)],              # Upper face (U)
                            1: [0, range(0, cube_size)],              # Down face (D)
                            4: [cube_size - 1, range(cube_size - 1, -1, -1)],  # Back face (B)
                            5: [0, range(0, cube_size)],              # Front face (F)
                        },
                        # Right face (R) -> adjacent faces: U, D, B, F
                        3: {
                            0: [cube_size - 1, range(0, cube_size)],  # Upper face (U)
                            1: [cube_size - 1, range(0, cube_size)],  # Down face (D)
                            4: [0, range(cube_size - 1, -1, -1)],     # Back face (B)
                            5: [cube_size - 1, range(0, cube_size)],  # Front face (F)
                        },
                        # Back face (B) -> adjacent faces: U, D, L, R
                        4: {
                            0: [range(0, cube_size), cube_size - 1],  # Upper face (U)
                            1: [range(cube_size - 1, -1, -1), 0],     # Down face (D)
                            2: [0, range(0, cube_size)],              # Left face (L)
                            3: [cube_size - 1, range(cube_size - 1, -1, -1)],  # Right face (R)
                        },
                        # Front face (F) -> adjacent faces: U, D, L, R
                        5: {
                            0: [range(0, cube_size), 0],              # Upper face (U)
                            1: [range(cube_size - 1, -1, -1), cube_size - 1],  # Down face (D)
                            2: [cube_size - 1, range(0, cube_size)],  # Left face (L)
                            3: [0, range(cube_size - 1, -1, -1)],     # Right face (R)
                        }}

            face = faces_to_colors[f][0]
            faces_to = adj_faces[face]
            if sign == 1:
                faces_from = faces_to[(np.arange(0, len(faces_to)) + 1) % len(faces_to)]
            else:
                faces_from = faces_to[(np.arange(len(faces_to) - 1, len(faces_to) - 1 + len(faces_to)))
                                      % len(faces_to)]

            cubes_idxs = [
                            [0, range(0, cube_size)],
                            [range(0, cube_size), cube_size - 1],
                            [cube_size - 1, range(cube_size - 1, -1, -1)],
                            [range(cube_size - 1, -1, -1), 0]
                        ]
            cubes_to = np.array([0, 1, 2, 3])

            if sign == 1:
                cubes_from = cubes_to[(np.arange(len(cubes_to) - 1, len(cubes_to) - 1 + len(cubes_to)))
                                      % len(cubes_to)]
            else:
                cubes_from = cubes_to[(np.arange(0, len(cubes_to)) + 1) % len(cubes_to)]

            for i in range(4):
                idxs_new = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_to[i]][0]]).flatten()
                            for idx2 in np.array([cubes_idxs[cubes_to[i]][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([cubes_idxs[cubes_from[i]][0]]).flatten()
                            for idx2 in np.array([cubes_idxs[cubes_from[i]][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face, idxNew[0], idxNew[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face, idxOld[0], idxOld[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))

            # Rotate adjacent faces
            face_idxs = adj_idxs[face]
            for i in range(0, len(faces_to)):
                face_to = faces_to[i]
                face_from = faces_from[i]
                idxs_new = [[idx1, idx2] for idx1 in np.array([face_idxs[face_to][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_to][1]]).flatten()]
                idxs_old = [[idx1, idx2] for idx1 in np.array([face_idxs[face_from][0]]).flatten() for idx2 in
                            np.array([face_idxs[face_from][1]]).flatten()]
                for idxNew, idxOld in zip(idxs_new, idxs_old):
                    flat_idx_new = np.ravel_multi_index((face_to, idxNew[0], idxNew[1]), colors_new.shape)
                    flat_idx_old = np.ravel_multi_index((face_from, idxOld[0], idxOld[1]), colors.shape)
                    rotate_idxs_new[move] = np.concatenate((rotate_idxs_new[move], [flat_idx_new]))
                    rotate_idxs_old[move] = np.concatenate((rotate_idxs_old[move], [flat_idx_old]))
        return rotate_idxs_new, rotate_idxs_old

    def _get_solved_state(self):
        return CubeState(np.array(np.arange(0, 6 * self.N**2), dtype=self.dtype))
