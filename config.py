# config.py

# Server Configuration
SERVER_HOST = '0.0.0.0'
PORT = 8080
DEBUG = False

# Model Configuration
MODEL_FILE = 'trained_models/3x3/maxback10/best_model_epoch_97.pt'

# Solver Configuration
MAX_SOLVE_TIME = 20
SOLVE_NUM_THREADS = 4

# API Configuration
API_VERSION = 'v1'
API_PREFIX = f'/api/{API_VERSION}'

# Cube Configuration
CUBE_SIZE = 3

# Client Configuration
SERVER_URL = "http://localhost:8080"  # Change this to your deployed server URL if needed
MAX_ALLOWED_SOLVE_TIME = 60
NUM_SCRAMBLES = 5
MAX_SCRAMBLES = 10

# Visualization Configuration
FIGURE_SIZE = (10, 12)
