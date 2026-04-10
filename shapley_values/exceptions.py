class CausalModelException(Exception):
    """
    Exception raised when a causal model is not properly defined.
    
    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message="Causal graph is not defined or is incomplete."):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message
