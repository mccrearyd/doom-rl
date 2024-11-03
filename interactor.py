import torch
import numpy as np
import gymnasium
from gymnasium.vector.utils import batch_space
import cv2
# from vizdoom import gymnasium_wrapper
# import doom

from custom_doom import VizDoomCustom

# from gymnasium.envs.registration import register


# register(
#     id="VizdoomOblige-v0",
#     entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
#     kwargs={"scenario_file": "oblige.cfg"},
# )


# for when we want to have the training live in a cv2 window
DISPLAY_SIZE = (1280, 720)


class VizDoomVectorized:
    def __init__(self, num_envs: int, env_id: str):
        self.num_envs = num_envs

        if env_id == "VizdoomCustom-v0":
            self.envs = [VizDoomCustom() for _ in range(num_envs)]
        else:
            self.envs = [gymnasium.make(env_id) for _ in range(num_envs)]
            
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

        all_infos = []

        for i in range(self.num_envs):
            if self.dones[i]:
                # Reset the environment if it was done in the last step
                obs, infos = self.envs[i].reset()
                self.observations[i] = torch.tensor(obs["screen"], dtype=torch.uint8)  # Fill the pre-allocated tensor
                self.rewards[i] = 0  # No reward on reset
                self.dones_tensor[i] = False
                self.dones[i] = False

                all_infos.append(infos)
            else:
                obs, reward, terminated, truncated, infos = self.envs[i].step(actions[i])
                self.observations[i] = torch.tensor(obs["screen"], dtype=torch.uint8)  # Fill the pre-allocated tensor
                self.rewards[i] = reward
                done = terminated or truncated
                self.dones_tensor[i] = done
                self.dones[i] = done

                all_infos.append(infos)

        return self.observations, self.rewards, self.dones_tensor, all_infos

    def close(self):
        for env in self.envs:
            env.close()

class DoomInteractor:
    """This thing manages the state of the environment and uses the agent
    to infer and step on the environment. This way is a bit easier
    because we can have environment mp while relying on the agent's
    internal vectorization, making gradients easier to accumulate.
    """

    def __init__(self, num_envs: int, watch: bool = False, env_id: str = "VizdoomCorridor-v0"):
        self.num_envs = num_envs
        self.env = VizDoomVectorized(num_envs, env_id=env_id)  # Using the vectorized environment
        self.single_action_space = self.env.envs[0].action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.watch = watch  # If True, OpenCV window will display frames from env 0
        self.watch_index = 0

        # OpenCV window for visualization
        if self.watch:
            cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("screen", *DISPLAY_SIZE)

    @property
    def avg_rew_per_frame(self):
        return self.current_episode_cumulative_rewards / (self.step_counter + 1)

    def reset(self):
        self.current_episode_cumulative_rewards = torch.zeros(self.num_envs, dtype=torch.float32)
        self.step_counter = torch.zeros(self.num_envs, dtype=torch.int32)

        return self.env.reset()

    def step(self, actions=None):
        if actions is None:
            actions = np.array([self.env.envs[i].action_space.sample() for i in range(self.num_envs)])

        # Step the environments with the sampled actions
        observations, rewards, dones, infos = self.env.step(actions)
        self.current_episode_cumulative_rewards += rewards
        self.step_counter += 1

        # Show the screen from the 0th environment if watch is enabled
        if self.watch:
            # Convert tensor to numpy array for OpenCV display
            screen = observations[self.watch_index].cpu().numpy()
            screen = cv2.resize(screen, DISPLAY_SIZE)

            # on the screen, draw the watch_index
            cv2.putText(screen, f"Env: {self.watch_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # also display the current reward
            cv2.putText(screen, f"Avg Reward per Frame: {self.avg_rew_per_frame[self.watch_index]:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("screen", screen)
            cv2.waitKey(1)  # Display for 1 ms

        # reset the reward sums for the environments that are done
        for i in range(self.num_envs):
            # if the average reward per frame falls below -1, let's reset
            if self.avg_rew_per_frame[i] < -1:
                new_obs, _ = self.env.envs[i].reset()
                observations[i] = torch.tensor(new_obs["screen"], dtype=torch.uint8)
                self.current_episode_cumulative_rewards[i] = 0
                self.step_counter[i] = 0

            if dones[i]:
                self.current_episode_cumulative_rewards[i] = 0
                self.step_counter[i] = 0

        # Return the results
        return observations, rewards, dones, infos

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
