# model_config.py
MODEL_DIR = "models/"
MODEL_WEIGHTS = "best_model_epoch_97.pt"
MODEL_FILE = MODEL_DIR + MODEL_WEIGHTS

# Cube parameters
CUBE_SIZE = 3  # 3x3 Rubik's Cube

# Model parameters
STATE_DIM = 6 * CUBE_SIZE**2  # 6 * 3 * 3 for a 3x3 Rubik's Cube
ONE_HOT_DEPTH = 6
H1_DIM = 5000
RESNET_DIM = 1000
NUM_RESNET_BLOCKS = 4
POLICY_OUT_DIM = 12
VALUE_OUT_DIM = 1
BATCH_NORM = True
DROPOUT = 0.1
