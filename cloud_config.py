import os


class CloudConfig:
    # Google Cloud Storage configuration
    GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', 'resnet_bucket_for_mcts')
    GCS_BLOB_NAME = os.environ.get('GCS_BLOB_NAME', 'best_model_epoch_97.pt')
    # Model configuration
    MODEL_CACHE_DIR = '/tmp/model_cache'
    MODEL_CACHE_TIME = 3600  # Cache the model for 1 hour

    # API configuration
    API_VERSION = 'v1'
    API_PREFIX = f'/api/{API_VERSION}'

    # Solver configuration
    MAX_SOLVE_TIME = 20
    SOLVE_NUM_THREADS = 4

    @classmethod
    def from_env(cls):
        """
        Create a CloudConfig instance from environment variables.
        This method allows for easy testing and configuration changes.
        """
        return cls()
