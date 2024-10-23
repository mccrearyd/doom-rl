import torch
import numpy as np
import gymnasium
from gymnasium.vector.utils import batch_space
from concurrent.futures import ThreadPoolExecutor
from vizdoom import gymnasium_wrapper


class VizDoomVectorized:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.envs = [gymnasium.make("VizdoomCorridor-v0") for _ in range(num_envs)]
        self.dones = [False] * num_envs

    def reset(self):
        observations = []
        for i in range(self.num_envs):
            obs, _ = self.envs[i].reset()
            observations.append(obs["screen"])  # Assuming we're focusing on the screen output
        return torch.tensor(observations, dtype=torch.uint8)  # Returning observations as a tensor

    def step(self, actions):
        """Steps all environments in parallel and returns observations, rewards, and done flags in tensors.
           If an environment is done, it will automatically reset.
        """
        observations = []
        rewards = []
        dones = []

        # Step each environment
        for i in range(self.num_envs):
            if self.dones[i]:
                # Reset the environment if it was done in the last step
                obs, _ = self.envs[i].reset()
                observations.append(obs["screen"])
                rewards.append(0)  # No reward on reset
                dones.append(False)
                self.dones[i] = False
            else:
                obs, reward, terminated, truncated, _ = self.envs[i].step(actions[i])
                observations.append(obs["screen"])
                rewards.append(reward)
                done = terminated or truncated
                dones.append(done)
                self.dones[i] = done

        return (
            torch.tensor(observations, dtype=torch.uint8),  # Return observations as a tensor
            torch.tensor(rewards, dtype=torch.float32),      # Return rewards as a tensor
            torch.tensor(dones, dtype=torch.bool)            # Return done flags as a tensor
        )

    def close(self):
        for env in self.envs:
            env.close()


class Interactor:
    """This thing manages the state of the environment and uses the agent
    to infer and step on the environment. This way is a bit easier
    because we can have environment mp while relying on the agent's
    internal vectorization, making gradients easier to accumulate.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.env = VizDoomVectorized(num_envs)  # Using the vectorized environment
        self.action_space = batch_space(self.env.envs[0].action_space, self.num_envs)

    def step(self):
        # Simulate actions by sampling from the action space
        actions = np.array([self.env.envs[i].action_space.sample() for i in range(self.num_envs)])

        # Step the environments with the sampled actions
        observations, rewards, dones = self.env.step(actions)

        # Return the results (you could also print or log them)
        return observations, rewards, dones


if __name__ == "__main__":
    num_envs = 1
    interactor = Interactor(num_envs)

    # Reset all environments
    observations = interactor.env.reset()
    print("Initial Observations:", observations.shape)

    # Example of stepping through the environments
    for _ in range(100):  # Step for 100 frames or episodes
        observations, rewards, dones = interactor.step()
        print("Observations Shape:", observations.shape)
        print("Rewards:", rewards)
        print("Dones:", dones)

    # Close all environments
    interactor.env.close()
