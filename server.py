import os
import torch
from flask import Flask, request, jsonify, Blueprint
from google.cloud import storage
from Environment import Cube
from MCTS.mcts import solve_cube
from utils.nnet_utils import get_device
from cloud_config import CloudConfig

app = Flask(__name__)
config = CloudConfig.from_env()

# Initialize Google Cloud Storage client
storage_client = storage.Client()

@app.route('/')
def home():
    return "Rubik's Cube Solver API is running!"

class CubeSolver:
    def __init__(self, cube_size=3):
        self.environment = Cube(cube_size)
        self.model = None
        self.model_weights = None

    def _load_model_weights(self):
        if self.model_weights is None:
            bucket = storage_client.bucket(config.GCS_BUCKET_NAME)
            blob = bucket.blob(config.GCS_BLOB_NAME)
            self.model_weights = blob.download_as_string()
        
        model = self.environment.get_nnet_model()
        model.load_state_dict(torch.load(self.model_weights, map_location=get_device()))
        self.model = model.eval()

    def solve(self, cube_state):
        self._load_model_weights()
        self.environment.set_state(cube_state)
        solution, complete = solve_cube(self.environment, self.model, self.environment.get_current_state(), 
                                        max_time=config.MAX_SOLVE_TIME, num_threads=config.SOLVE_NUM_THREADS)
        return solution, complete

solver = CubeSolver()

# Create a Blueprint for API versioning
api = Blueprint('api', __name__, url_prefix=config.API_PREFIX)

@api.route('/solve', methods=['POST'])
def solve():
    data = request.json
    cube_state = data.get('cube_state')
    
    if not cube_state:
        return jsonify({'error': 'Missing cube_state in request'}), 400

    try:
        solution, complete = solver.solve(cube_state)
    except Exception as e:
        app.logger.error(f"Error solving cube: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

    if complete:
        app.logger.info(f"Solution found: {len(solution)} moves")
        move_names = [f"{solver.environment.moves[move][0]}{'' if solver.environment.moves[move][1] == 1 else '`'}" for move in solution]
        solution_str = ' '.join(move_names)
    else:
        app.logger.warning("Could not find a complete solution in the given time.")
        solution_str = "No solution found"

    return jsonify({
        'solution': solution_str,
        'complete': complete
    })

# Register the Blueprint
app.register_blueprint(api)

if __name__ == '__main__':
    print(f"Server is running on http://localhost:{int(os.environ.get('PORT', 8080))}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)