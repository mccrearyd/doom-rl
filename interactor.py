import torch
import numpy as np
import gymnasium
from gymnasium.vector.utils import batch_space
import cv2
from vizdoom import gymnasium_wrapper

import doom

# from gymnasium.envs.registration import register


# register(
#     id="VizdoomOblige-v0",
#     entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
#     kwargs={"scenario_file": "oblige.cfg"},
# )


DISPLAY_SIZE = (1280, 720)


class VizDoomVectorized:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.envs = [gymnasium.make("VizdoomOblige-v0") for _ in range(num_envs)]
        self.dones = [False] * num_envs

        # Pre-allocate observation and reward tensors
        first_obs_space = self.envs[0].observation_space['screen']
        self.obs_shape = first_obs_space.shape
        self.observations = torch.zeros((num_envs, *self.obs_shape), dtype=torch.uint8)
        self.rewards = torch.zeros(num_envs, dtype=torch.float32)
        self.dones_tensor = torch.zeros(num_envs, dtype=torch.bool)

    def reset(self):
        for i in range(self.num_envs):
            obs, _ = self.envs[i].reset()
            self.observations[i] = torch.tensor(obs["screen"], dtype=torch.uint8)  # Fill the pre-allocated tensor
            self.dones[i] = False
        return self.observations

    def step(self, actions):
        """Steps all environments in parallel and fills pre-allocated tensors for observations, rewards, and dones.
           If an environment is done, it will automatically reset.
        """
        for i in range(self.num_envs):
            if self.dones[i]:
                # Reset the environment if it was done in the last step
                obs, _ = self.envs[i].reset()
                self.observations[i] = torch.tensor(obs["screen"], dtype=torch.uint8)  # Fill the pre-allocated tensor
                self.rewards[i] = 0  # No reward on reset
                self.dones_tensor[i] = False
                self.dones[i] = False
            else:
                obs, reward, terminated, truncated, _ = self.envs[i].step(actions[i])
                self.observations[i] = torch.tensor(obs["screen"], dtype=torch.uint8)  # Fill the pre-allocated tensor
                self.rewards[i] = reward
                done = terminated or truncated
                self.dones_tensor[i] = done
                self.dones[i] = done

        return self.observations, self.rewards, self.dones_tensor

    def close(self):
        for env in self.envs:
            env.close()

class DoomInteractor:
    """This thing manages the state of the environment and uses the agent
    to infer and step on the environment. This way is a bit easier
    because we can have environment mp while relying on the agent's
    internal vectorization, making gradients easier to accumulate.
    """

    def __init__(self, num_envs: int, watch: bool = False):
        self.num_envs = num_envs
        self.env = VizDoomVectorized(num_envs)  # Using the vectorized environment
        self.action_space = batch_space(self.env.envs[0].action_space, self.num_envs)
        self.watch = watch  # If True, OpenCV window will display frames from env 0

        # OpenCV window for visualization
        if self.watch:
            cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("screen", *DISPLAY_SIZE)

    def reset(self):
        return self.env.reset()

    def step(self, actions=None):
        if actions is None:
            actions = np.array([self.env.envs[i].action_space.sample() for i in range(self.num_envs)])

        # Step the environments with the sampled actions
        observations, rewards, dones = self.env.step(actions)

        # Show the screen from the 0th environment if watch is enabled
        if self.watch:
            # Convert tensor to numpy array for OpenCV display
            screen = observations[0].cpu().numpy()
            screen = cv2.resize(screen, DISPLAY_SIZE)

            cv2.imshow("screen", screen)
            cv2.waitKey(1)  # Display for 1 ms

        # Return the results
        return observations, rewards, dones

    def close(self):
        if self.watch:
            cv2.destroyAllWindows()  # Close the OpenCV window
        self.env.close()


if __name__ == "__main__":
    MAX_STEPS = 100
    NUM_ENVS = 16
    
    # if true one of the environments will be displayed in a cv2 window
    WATCH = False
    
    interactor = DoomInteractor(NUM_ENVS, watch=WATCH)

    # Reset all environments
    observations = interactor.env.reset()
    # print("Initial Observations:", observations.shape)

    # Example of stepping through the environments
    for _ in range(100):  # Step for 100 frames or episodes

        observations, rewards, dones = interactor.step()
        print(observations.shape, rewards.shape)

    # Close all environments
    interactor.env.close()
