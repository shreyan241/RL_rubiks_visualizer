import numpy as np
import traceback
from flask import Flask, request, jsonify, Blueprint
from Environment import Cube
from MCTS.mcts import solve_cube
from utils.nnet_utils import get_device
from Environment import CubeState, move_dict
import config

app = Flask(__name__)


class CubeSolver:
    def __init__(self, cube_size=config.CUBE_SIZE):
        self.environment = Cube(N=cube_size)
        self.model = None

    def _load_model(self):
        if self.model is None:
            print("Loading model...")
            model = self.environment.get_nnet_model()
            device = get_device()
            self.model = self.environment.load_nnet_model(config.MODEL_FILE, model, device)
            print("Model loaded successfully")

    def solve(self, cube_state, max_time):
        print(f"Solving cube state: {cube_state}")
        self._load_model()
        cube_state_obj = CubeState(np.array(cube_state, dtype=self.environment.dtype))
        cube_to_solve = Cube(N=self.environment.N, current_state=cube_state_obj)
        print(f"Starting MCTS solve with max time: {max_time}")
        solution, complete = solve_cube(cube_to_solve, self.model, cube_to_solve.current_state,
                                        max_time=max_time, num_threads=config.SOLVE_NUM_THREADS)
        print(f"MCTS solve complete. Solution: {solution}, Complete: {complete}")
        return solution, complete


solver = CubeSolver()
api = Blueprint('api', __name__, url_prefix=config.API_PREFIX)


@api.route('/solve', methods=['POST'])
def solve():
    try:
        data = request.get_json()
        if not data or 'cube_state' not in data:
            return jsonify({'error': 'Invalid input. cube_state is required.'}), 400

        cube_state = data['cube_state']
        max_time = min(data.get('max_time', config.MAX_SOLVE_TIME), config.MAX_SOLVE_TIME)

        print(f"Received cube state: {cube_state}")
        print(f"Max solve time: {max_time}")

        solution, complete = solver.solve(cube_state, max_time)
        print(f"Solution found: {solution}, Complete: {complete}")

        if complete:
            move_names = [f"{move_dict[move][0]}{'' if move_dict[move][1] == 1 else '`'}" for move in solution]
            solution_str = ' '.join(move_names)
            return jsonify({
                'solution': solution_str,
                'complete': complete,
                'num_moves': len(solution)
            })
        else:
            return jsonify({
                'solution': 'No solution found',
                'complete': complete
            })
    except Exception as e:
        error_msg = f"Error in solve endpoint: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500


app.register_blueprint(api)


@app.route('/')
def home():
    return "Rubik's Cube Solver API is running!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.PORT, debug=config.DEBUG)
