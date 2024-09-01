from typing import Dict, List, Tuple, Union
import numpy as np

from Environment.CubeConfig import adj_faces, faces_to_colors


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

    def __init__(self, N=3):
        super().__init__()
        self.dtype = np.uint8
        self.N = N
        self.solvedState = self._get_solved_state()
        self.current_state = CubeState(self.solvedState.copy())
        self.moves: List[Tuple[str, int]] = [(a, t) for a in ['U', 'D', 'L', 'R', 'B', 'F'] for t in [1, -1]]
        self.moves_rev: List[Tuple[str, int]] = [(a, t) for a in ['U', 'D', 'L', 'R', 'B', 'F'] for t in [-1, 1]]

        # get idxs changed for moves
        self.rotate_idxs_new: Dict[Tuple[str, int], np.ndarray] = dict()
        self.rotate_idxs_old: Dict[Tuple[str, int], np.ndarray] = dict()
        self.rotate_idxs_new, self.rotate_idxs_old = self._precompute_rotation_idxs(self.N, self.moves)

    def next_state(self, states: List[CubeState], move: int) -> Tuple[List[CubeState], List[float]]:
        states_np = np.stack([x.colors for x in states], axis=0)
        states_next_np, transition_costs = self._move_np(states_np, move)

        states_next: List[CubeState] = [CubeState(x) for x in list(states_next_np)]

        return states_next, transition_costs

    def prev_state(self, states: List[CubeState], move: int) -> List[CubeState]:
        move: Tuple[str, int] = self.moves[move]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def apply_move(self, move: int):
        """Applies a move to the current state of the cube."""
        next_states, _ = self.next_state([self.current_state], move)
        self.current_state = next_states[0]  # Update the current state

    def get_current_state(self) -> CubeState:
        """Returns the current state of the cube."""
        return self.current_state

    def generate_solved_states(self, num_states: int, np_format: bool = False) -> Union[
                                                                                List[CubeState], np.ndarray]:
        if np_format:
            solved_np: np.ndarray = np.expand_dims(self.solvedState.copy(), 0)
            solved_states: np.ndarray = np.repeat(solved_np, num_states, axis=0)
        else:
            solved_states: List[CubeState] = [CubeState(self.solvedState.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[CubeState]) -> np.ndarray:
        states_np = np.stack([state.colors for state in states], axis=0)
        is_equal = np.equal(states_np, np.expand_dims(self.solvedState, 0))

        return np.all(is_equal, axis=1)

    def get_num_moves(self) -> int:
        return len(self.moves)

    def expand(self, states: List[CubeState]) -> Tuple[List[List[CubeState]], List[np.ndarray]]:

        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[CubeState]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # numpy states
        states_np: np.ndarray = np.stack([state.colors for state in states])

        # for each move, get next states, transition costs, and if solved
        move_idx: int
        # move: int
        for move_idx in range(num_env_moves):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_np(states_np, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(CubeState(states_next_np[idx]))

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l

    def _move_np(self, states_np: np.ndarray, action: int):
        action_key: tuple[str, int] = self.moves[action]

        states_next_np: np.ndarray = states_np.copy()
        states_next_np[:, self.rotate_idxs_new[action_key]] = states_np[:, self.rotate_idxs_old[action_key]]

        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]
        return states_next_np, transition_costs

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
        return np.array(np.arange(0, 6 * self.N**2), dtype=self.dtype)
