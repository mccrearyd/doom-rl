


class Interactor:
    """This thing manages the state of the environment and uses the agent
    to infer and step on the environment. This way is a bit easier
    because we can have environment mp while relying on the agent's
    internal vectorization, making gradients easier to accumulate.
    """

    def __init__(self):
        pass