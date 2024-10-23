import torch
import numpy as np

import gymnasium
from gymnasium.vector.utils import batch_space


class VizDoomVectorized:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    def reset(self):
        # TODO: reset all envs and return the observations in one tensor
        pass

    def step(self, actions):
        # TODO: this method should accept the actions and return the observations and rewards in two tensors.
        # it should step the environments in parallel. we will be calling step() as many times as we can in a
        # vectorized episode.

        # could you also make it such that if the environment is done, it resets itself?
        pass
    


class Interactor:
    """This thing manages the state of the environment and uses the agent
    to infer and step on the environment. This way is a bit easier
    because we can have environment mp while relying on the agent's
    internal vectorization, making gradients easier to accumulate.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs

        self.env = None  # TODO
        self.action_space = batch_space(self.env.action_space, self.num_envs)

    def step(self):
        # for now, let's just use action_space.sample()

        observations = torch.zeros(size=(self.num_envs, 240, 320, 3), dtype=torch.uint8)

        # HACK: pretend like we do an agent.forward(observations)
        actions = self.env.action_space.sample()

        