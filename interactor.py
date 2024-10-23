import ray
import torch
import numpy as np
import gymnasium
from gymnasium.vector.utils import batch_space
import cv2
from vizdoom import gymnasium_wrapper

# Initialize ray
ray.init(ignore_reinit_error=True)

@ray.remote
class DoomWorker:
    def __init__(self, num_envs: int, watch: bool = False):
        self.num_envs = num_envs
        self.env = VizDoomVectorized(num_envs)  # Using the vectorized environment
        self.action_space = batch_space(self.env.envs[0].action_space, self.num_envs)
        self.watch = watch  # If True, OpenCV window will display frames from env 0

        if self.watch:
            cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("screen", 640, 480)

    def reset(self):
        return self.env.reset()

    def step(self, actions=None):
        if actions is None:
            actions = np.array([self.env.envs[i].action_space.sample() for i in range(self.num_envs)])

        observations, rewards, dones = self.env.step(actions)

        if self.watch:
            screen = observations[0].cpu().numpy()
            cv2.imshow("screen", screen)
            cv2.waitKey(1)  # Display for 1 ms

        return observations, rewards, dones

    def close(self):
        if self.watch:
            cv2.destroyAllWindows()
        self.env.close()


class DoomInteractor:
    """This class manages the state of the environment using Ray's remote workers.
    It allows parallelization across both multiple workers and multiple environments.
    """

    def __init__(self, num_workers: int, num_envs: int, watch: bool = False):
        self.k_workers = num_workers
        self.num_envs_per_worker = num_envs

        # Initialize `k_workers` remote workers, each with `num_envs_per_worker` environments
        self.workers = [DoomWorker.remote(num_envs, watch) for _ in range(num_workers)]

    def reset(self):
        # Call reset on all workers asynchronously
        return ray.get([worker.reset.remote() for worker in self.workers])

    def step(self, actions=None):
        # Call step on all workers asynchronously
        if actions is None:
            results = [worker.step.remote() for worker in self.workers]
        else:
            results = [worker.step.remote(actions[i]) for i, worker in enumerate(self.workers)]
        
        return ray.get(results)

    def close(self):
        # Call close on all workers
        ray.get([worker.close.remote() for worker in self.workers])


class VizDoomVectorized:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.envs = [gymnasium.make("VizdoomCorridor-v0") for _ in range(num_envs)]
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
        for i in range(self.num_envs):
            if self.dones[i]:
                obs, _ = self.envs[i].reset()
                self.observations[i] = torch.tensor(obs["screen"], dtype=torch.uint8)
                self.rewards[i] = 0
                self.dones_tensor[i] = False
                self.dones[i] = False
            else:
                obs, reward, terminated, truncated, _ = self.envs[i].step(actions[i])
                self.observations[i] = torch.tensor(obs["screen"], dtype=torch.uint8)
                self.rewards[i] = reward
                done = terminated or truncated
                self.dones_tensor[i] = done
                self.dones[i] = done

        return self.observations, self.rewards, self.dones_tensor

    def close(self):
        for env in self.envs:
            env.close()


if __name__ == "__main__":
    MAX_STEPS = 100
    K_WORKERS = 4
    NUM_ENVS_PER_WORKER = 4  # Total environments = K_WORKERS * NUM_ENVS_PER_WORKER
    WATCH = False

    interactor = DoomInteractor(K_WORKERS, NUM_ENVS_PER_WORKER, watch=WATCH)

    # Reset all environments across all workers
    observations = interactor.reset()

    # Example of stepping through the environments in parallel
    for step in range(MAX_STEPS):
        results = interactor.step()
        observations, rewards, dones = zip(*results)

        # Combine all results into single tensors
        observations = torch.cat(observations)
        rewards = torch.cat(rewards)
        dones = torch.cat(dones)

        print(f"Step {step}: Observations shape: {observations.shape}, Rewards: {rewards.sum()}, Dones: {dones.sum()}")

    # Close all environments
    interactor.close()

    # Shutdown ray
    ray.shutdown()
