import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from threading import Thread, Lock
import time
from Environment.cube import Cube, CubeState
# from Environment.CubeConfig import move_to_rev_dict
# from utils.nnet_utils import get_device


class MCTSNode:
    def __init__(self, state: CubeState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[int, MCTSNode] = {}
        self.N: Dict[int, int] = {}  # visit count
        self.W: Dict[int, float] = {}  # max value
        self.L: Dict[int, float] = {}  # virtual loss
        self.P: Dict[int, float] = {}  # prior probabilities
        self.is_fully_expanded = False


class MCTS:
    def __init__(self, cube: Cube, model: nn.Module, c: float = 2.0, nu: float = 0.1):
        self.cube = cube
        self.model = model
        self.c = c  # exploration parameter
        self.nu = nu  # virtual loss parameter
        self.lock = Lock()  # Add a lock for thread synchronization
        self.best_solution = None
        self.best_solution_length = float('inf')
        self.complete_solution_found = False

    def search(self, root_state: CubeState, max_time: float = 5.0, num_threads: int = 4) -> List[int]:
        root = MCTSNode(root_state)
        self.expand(root)

        def worker(worker_id):
            local_best_solution = None
            local_best_solution_length = float('inf')

            while time.time() - start_time < max_time:
                leaf = self.select(root)
                if self.cube.is_solved([leaf.state])[0]:
                    solution = self.extract_solution(leaf)
                    if len(solution) < local_best_solution_length:
                        local_best_solution = solution
                        local_best_solution_length = len(solution)
                        with self.lock:
                            if local_best_solution_length < self.best_solution_length:
                                self.best_solution = local_best_solution
                                self.best_solution_length = local_best_solution_length
                                self.complete_solution_found = True
                        # print(f"Worker {worker_id} found a complete solution of length {len(solution)}!")
                    continue  # Continue searching for better solutions
                self.expand(leaf)
                value = self.evaluate(leaf)
                self.backpropagate(leaf, value)

        start_time = time.time()
        threads = [Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # if self.complete_solution_found:
        #     print(f"Best complete solution found with length {self.best_solution_length}")
        # else:
        #     print("Search timed out. Returning best partial solution found.")

        return self.best_solution if self.best_solution else self.extract_best_path(root), self.complete_solution_found

    def select(self, node: MCTSNode) -> MCTSNode:
        path = []
        while node.children:
            if not node.is_fully_expanded:
                return self.expand(node)
            action = self.select_action(node)
            path.append(action)
            node = node.children[action]
        return node

    def select_action(self, node: MCTSNode) -> int:
        total_visits = sum(node.N.values())

        def U(a):
            return self.c * node.P[a] * np.sqrt(total_visits) / (1 + node.N[a])

        def Q(a):
            return node.W[a] - node.L[a]

        return max(node.children.keys(), key=lambda a: U(a) + Q(a))

    def expand(self, node: MCTSNode):
        # Acquire lock to ensure only one worker expands this node at a time
        with self.lock:
            if node.is_fully_expanded:  # Node is already fully expanded, so skip it
                return

            if not node.children:
                next_states = self.cube.expand([node.state])[0]  # Expand into child states
                policy, value = self.predict(node.state)  # Get policy (and value) using the neural network

                for action, next_state in enumerate(next_states):
                    if action not in node.children:
                        child = MCTSNode(next_state, parent=node, action=action)
                        node.children[action] = child
                        node.N[action] = 0
                        node.W[action] = 0
                        node.L[action] = 0
                        node.P[action] = policy[action]  # Set prior probability for each action
            node.is_fully_expanded = True

    def evaluate(self, node: MCTSNode) -> float:
        _, value = self.predict(node.state)
        return value

    def backpropagate(self, node: MCTSNode, value: float):
        with self.lock:
            while node.parent:
                action = node.action
                node = node.parent
                node.N[action] += 1
                node.W[action] = max(node.W[action], value)  # Max value is propagated
                node.L[action] = max(0, node.L[action] - self.nu)  # Virtual loss reduced

    def predict(self, state: CubeState) -> Tuple[np.ndarray, float]:
        with torch.no_grad():
            # Use the cube's state-to-network input function
            input_tensor = torch.FloatTensor(self.cube.state_to_nnet_input([state])[0])
            policy, value = self.model(input_tensor)
        return policy.numpy().flatten(), value.item()

    def extract_solution(self, node: MCTSNode) -> List[int]:
        actions = []
        while node.parent:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))  # Reverse the actions to get the correct solution sequence

    def extract_best_path(self, root: MCTSNode) -> List[int]:
        node = root
        actions = []
        while node.children:
            action = max(node.children.keys(), key=lambda a: node.N[a])  # Choose action with highest visit count
            actions.append(action)
            node = node.children[action]
        return actions


def solve_cube(cube: Cube, model: nn.Module, initial_state: CubeState, max_time: float = 20, num_threads: int = 4) -> List[int]:
    mcts = MCTS(cube, model)
    solution, complete = mcts.search(initial_state, max_time, num_threads)
    return solution, complete


# if __name__ == "__main__":
#     cube = Cube(N=3)  # 3x3 Rubik's Cube

#     # Initialize the model
#     model = cube.get_nnet_model()  # Create a new model instance

#     # Load the model weights from file
#     model_file = r"trained_models\3x3\best_model_epoch_97.pt"
#     device = get_device()
#     model = cube.load_nnet_model(model_file, model, device)
#     # Now you can use the loaded model in your MCTS algorithm
#     initial_state, random_moves = cube.scramble_states([cube.solvedState], num_scrambles=10)
#     solution, complete = solve_cube(cube, model, initial_state[0])
#     if complete:
#         print(f"Complete solution found: {solution}")
#     else:
#         print(f"Partial solution found (search timed out): {solution}")
#     true_sol = []
#     for random_move in random_moves[0]:
#         true_sol.append(move_to_rev_dict[random_move])
#     true_sol.reverse()
#     print("True solution:", true_sol)
