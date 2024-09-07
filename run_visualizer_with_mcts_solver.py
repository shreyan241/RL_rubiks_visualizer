import matplotlib.pyplot as plt
from Visualizer import RubiksCubeVisualizer, InteractiveCube
from Environment import Cube
from Environment.CubeConfig import move_dict
from MCTS.mcts import solve_cube
from utils.nnet_utils import get_device
from threading import Lock


class MctsSolverMixin:
    def mcts_solver_wrapper(self, max_time=20, num_threads=4):
        cube = self.environment
        solution, complete = solve_cube(cube, self.model, cube.get_current_state(), max_time=max_time, num_threads=num_threads)
        return solution, complete


class RubiksCubeVisualizerWithMcts(RubiksCubeVisualizer, MctsSolverMixin):
    def __init__(self, cube_env=Cube(3), N=3, plastic_color=None, face_colors=None):
        super().__init__(cube_env, N, plastic_color, face_colors)
        self.model = None

    def _load_model(self):
        if self.model is None:
            model = self.environment.get_nnet_model()
            model_file = r"trained_models\3x3\maxback10\best_model_epoch_97.pt"
            device = get_device()
            self.model = self.environment.load_nnet_model(model_file, model, device)
        return self.model

    def mcts_solver_wrapper(self, max_time=20, num_threads=4):
        self._load_model()  # Ensure model is loaded
        return super().mcts_solver_wrapper(max_time, num_threads)


class InteractiveCubeWithMcts(InteractiveCube):
    def __init__(self, visualizer, *args, **kwargs):
        super().__init__(visualizer, *args, **kwargs)
        self._solution_text = self.figure.text(0.05, 0.95, '', fontsize=10)
        self._current_move_text = self.figure.text(0.05, 0.90, '', fontsize=10, color='red')
        self._solve_lock = Lock()

    def _initialize_widgets(self):
        super()._initialize_widgets()
        self._btn_solve.on_clicked(self._solve_cube_wrapper)

    def _solve_cube_wrapper(self, event=None):
        if self._solve_lock.acquire(blocking=False):
            try:
                if not hasattr(self, "_solved"):  # Check if the solution has already been found
                    self._solve_cube()
                    self._solved = True  # Mark as solved so it doesn't repeat
            finally:
                self._solve_lock.release()

    def _solve_cube(self, event=None):
        print("Solving cube...")
        solution, complete = self.visualizer.mcts_solver_wrapper()

        if complete:
            print(f"Solution found: {len(solution)} moves")
            self._display_solution(solution)
            self._play_solution(solution)
        else:
            print("Could not find a complete solution in the given time.")
            self._display_no_solution()

        self._draw_cube()

    def _display_solution(self, solution):
        move_names = [f"{move_dict[move][0]}{'' if move_dict[move][1] == 1 else '`'}" for move in solution]
        solution_str = ' '.join(move_names)
        self._solution_text.set_text(f"Solution: {solution_str}")
        self._current_move_text.set_text('')
        self.figure.canvas.draw()

    def _display_no_solution(self):
        self._solution_text.set_text("No solution found in the given time.")
        self._current_move_text.set_text('')
        self.figure.canvas.draw()

    def _play_solution(self, solution):
        for i, move in enumerate(solution):
            face, turns = move_dict[move]
            self.rotate_face(face, turns)
            self._highlight_move(solution, i)
            self.figure.canvas.draw()
            plt.pause(0.5)  # Add a delay between moves for visibility

    def _highlight_move(self, solution, current_move):
        move_names = [f"{move_dict[move][0]}{'' if move_dict[move][1] == 1 else '`'}" for move in solution]
        highlighted_solution = [
            f"[{move}]" if i == current_move else move
            for i, move in enumerate(move_names)
        ]
        solution_str = ' '.join(highlighted_solution)
        self._solution_text.set_text(f"Solution: {solution_str}")

        current_move_name = move_names[current_move]
        self._current_move_text.set_text(f"Executing: {current_move_name}")

        self.figure.canvas.draw()


if __name__ == '__main__':
    import sys
    try:
        N = int(sys.argv[1])
        cube = Cube(N)
    except:
        N = 3
        cube = Cube(N)

    c = RubiksCubeVisualizerWithMcts(cube, N)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_axes(InteractiveCubeWithMcts(c))
    plt.show()
