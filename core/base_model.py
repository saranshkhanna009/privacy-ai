class BaseGenerativeModel:
    """Base class for generative models.
    Handles config storage and training flag.
    """
    def __init__(self, config: dict):
        self.config = config
        self.is_trained = False
