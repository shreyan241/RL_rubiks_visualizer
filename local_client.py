import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import numpy as np
import requests
from Visualizer import RubiksCubeVisualizer, InteractiveCube
from Environment import Cube


class RefinedInteractiveCube(InteractiveCube):
    def __init__(self, visualizer, server_url, *args, **kwargs):
        super().__init__(visualizer, *args, **kwargs)
        self.server_url = server_url
        self.max_solve_time = 20
        self.max_allowed_solve_time = 90
        self.num_scrambles = 5
        self.max_scrambles = 10
        self.solving = False
        self._setup_refined_interface()

    def _initialize_widgets(self):
        # Override to prevent original buttons from appearing
        pass

    def _setup_refined_interface(self):
        self.figure.set_size_inches(10, 12)

        # Stylized heading
        self.figure.text(0.5, 0.95, 'Reinforcement Learning Rubik\'s Cube Solver',
                         fontsize=18, fontweight='bold', ha='center', va='center')

        # Cube display area
        self.set_position([0.1, 0.30, 0.8, 0.6])

        # Input fields with individual set buttons
        self.ax_solve_time = self.figure.add_axes([0.7, 0.85, 0.1, 0.03])
        self.tb_solve_time = TextBox(self.ax_solve_time, 'Max Solve Time (seconds): ', initial=str(self.max_solve_time))
        self.ax_set_solve_time = self.figure.add_axes([0.81, 0.85, 0.08, 0.03])
        self.btn_set_solve_time = Button(self.ax_set_solve_time, 'Set')
        self.btn_set_solve_time.on_clicked(self._set_solve_time)

        self.ax_num_scrambles = self.figure.add_axes([0.7, 0.80, 0.1, 0.03])
        self.tb_num_scrambles = TextBox(self.ax_num_scrambles, f'Scrambles (max {self.max_scrambles}): ',
                                        initial=str(self.num_scrambles))
        self.ax_set_num_scrambles = self.figure.add_axes([0.81, 0.80, 0.08, 0.03])
        self.btn_set_num_scrambles = Button(self.ax_set_num_scrambles, 'Set')
        self.btn_set_num_scrambles.on_clicked(self._set_num_scrambles)

        # Solution display
        self._solution_text = self.figure.text(0.5, 0.25, '', fontsize=10, ha='center', va='center')
        self._current_move_text = self.figure.text(0.5, 0.20, '', fontsize=10, color='red', ha='center', va='center')

        # Feedback display
        self._feedback_text = self.figure.text(0.5, 0.05, '', fontsize=10, ha='center', va='center', color='blue')

        # Buttons (below cube, larger)
        button_width = 0.25
        button_height = 0.06
        self._ax_reset = self.figure.add_axes([0.1, 0.1, button_width, button_height])
        self._btn_reset = Button(self._ax_reset, 'Reset Cube')
        self._btn_reset.on_clicked(self._reset_cube_wrapper)

        self._ax_scramble = self.figure.add_axes([0.375, 0.1, button_width, button_height])
        self._btn_scramble = Button(self._ax_scramble, 'Scramble Cube')
        self._btn_scramble.on_clicked(self._scramble_cube_wrapper)

        self._ax_solve = self.figure.add_axes([0.65, 0.1, button_width, button_height])
        self._btn_solve = Button(self._ax_solve, 'Solve Cube')
        self._btn_solve.on_clicked(self._solve_cube_wrapper)

        # Override the base class's key press event
        self.figure.canvas.mpl_connect('key_press_event', self._on_key_press)

    def _clear_text_displays(self):
        self._solution_text.set_text('')
        self._current_move_text.set_text('')
        self._feedback_text.set_text('')
        self.figure.canvas.draw()

    def _set_solve_time(self, event=None):
        self._clear_text_displays()
        try:
            new_time = int(self.tb_solve_time.text)
            if 0 < new_time <= self.max_allowed_solve_time:
                self.max_solve_time = new_time
                feedback = f"Max solve time set to: {self.max_solve_time} seconds"
            elif new_time > self.max_allowed_solve_time:
                self.max_solve_time = self.max_allowed_solve_time
                feedback = f"Max solve time cannot be more than {self.max_allowed_solve_time} seconds. Set to maximum: {self.max_allowed_solve_time} seconds"
                self.tb_solve_time.set_val(str(self.max_solve_time))
            else:
                feedback = "Max solve time must be a positive integer. No change made."
                self.tb_solve_time.set_val(str(self.max_solve_time))
        except ValueError:
            feedback = "Invalid input for max solve time. Please enter a positive integer. No change made."
            self.tb_solve_time.set_val(str(self.max_solve_time))
        self._feedback_text.set_text(feedback)
        self.figure.canvas.draw()

    def _set_num_scrambles(self, event=None):
        self._clear_text_displays()
        try:
            new_scrambles = int(self.tb_num_scrambles.text)
            if new_scrambles > 0:
                if new_scrambles > self.max_scrambles:
                    self.num_scrambles = self.max_scrambles
                    feedback = f"Number of scrambles cannot be more than {self.max_scrambles}. Set to maximum: {self.max_scrambles}"
                else:
                    self.num_scrambles = new_scrambles
                    feedback = f"Number of scrambles set to: {self.num_scrambles}"
            else:
                feedback = "Number of scrambles must be a positive integer. No change made."
                self.tb_num_scrambles.set_val(str(self.num_scrambles))
        except ValueError:
            feedback = "Invalid input for number of scrambles. Please enter a positive integer. No change made."
            self.tb_num_scrambles.set_val(str(self.num_scrambles))
        self._feedback_text.set_text(feedback)
        self.figure.canvas.draw()

    def _scramble_cube_mcts(self, event=None):
        self._clear_text_displays()
        moves = np.random.choice(['U', 'D', 'L', 'R', 'F', 'B'], self.num_scrambles)
        for move in moves:
            self.rotate_face(move, turns=np.random.choice([-1, 1]))
        self._draw_cube()
        self._feedback_text.set_text(f"Cube scrambled with {self.num_scrambles} moves")
        self.figure.canvas.draw()

    def _solve_cube(self, event=None):
        self._clear_text_displays()
        if self.solving:
            return
        self.solving = True

        self._feedback_text.set_text("Solving cube...")
        self.figure.canvas.draw()

        cube_state = self.visualizer.environment.get_current_state().colors.tolist()
        try:
            response = requests.post(f"{self.server_url}/api/v1/solve",
                                     json={"cube_state": cube_state,
                                           "max_time": self.max_solve_time},
                                     timeout=self.max_solve_time + 5)

            if response.status_code == 200:
                data = response.json()
                if data['complete']:
                    print(f"Solution found: {data['num_moves']} moves")
                    self._display_solution(data['solution'])
                    self._play_solution(data['solution'])
                else:
                    print("Could not find a complete solution in the given time.")
                    self._display_no_solution()
            else:
                print(f"Error: {response.status_code}")
                print(f"Error message: {response.text}")
                self._display_no_solution()
        except requests.RequestException as e:
            print(f"Request failed: {str(e)}")
            self._display_no_solution()
        finally:
            self.solving = False

    def _display_solution(self, solution_str):
        self._solution_text.set_text(f"Solution: {solution_str}")
        self._current_move_text.set_text('')
        self._feedback_text.set_text("Solution found and displayed")
        self.figure.canvas.draw()

    def _display_no_solution(self):
        self._solution_text.set_text("No solution found in the given time.")
        self._current_move_text.set_text('')
        self._feedback_text.set_text("No solution found")
        self.figure.canvas.draw()

    def _play_solution(self, solution_str):
        moves = solution_str.split()
        for i, move_str in enumerate(moves):
            face = move_str[0]
            turns = -1 if move_str.endswith('`') else 1
            self.rotate_face(face, turns)
            self._highlight_move(moves, i)
            self.figure.canvas.draw()
            plt.pause(0.5)

    def _highlight_move(self, moves, current_move):
        highlighted_solution = [
            f"[{move}]" if i == current_move else move
            for i, move in enumerate(moves)
        ]
        solution_str = ' '.join(highlighted_solution)
        self._solution_text.set_text(f"Solution: {solution_str}")
        self._current_move_text.set_text(f"Executing: {moves[current_move]}")
        self.figure.canvas.draw()

    def _reset_cube_wrapper(self, event=None):
        self._clear_text_displays()
        super()._reset_cube(event)
        self._feedback_text.set_text("Cube reset to solved state")
        self.figure.canvas.draw()

    def _scramble_cube_wrapper(self, event=None):
        self._clear_text_displays()
        self._scramble_cube_mcts(event)

    def _solve_cube_wrapper(self, event=None):
        self._clear_text_displays()
        self._solve_cube(event)

    def _on_key_press(self, event):
        self._clear_text_displays()
        super()._on_key_press(event)
        if event.key in ['u', 'd', 'l', 'r', 'f', 'b', 'U', 'D', 'L', 'R', 'F', 'B']:
            self._feedback_text.set_text(f"Face rotated: {event.key.upper()}")
            self.figure.canvas.draw()


if __name__ == '__main__':
    server_url = "http://localhost:8080"  # Change this to your deployed server URL if needed
    cube = Cube(3)
    c = RubiksCubeVisualizer(cube, 3)
    fig = plt.figure(figsize=(10, 12))
    ax = fig.add_axes(RefinedInteractiveCube(c, server_url))
    plt.show()
