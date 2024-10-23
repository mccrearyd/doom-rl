import gymnasium
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from vizdoom import gymnasium_wrapper

def run_single_episode(env_id, max_frames):
    env = gymnasium.make("VizdoomCorridor-v0")
    observation, info = env.reset()
    rewards = []

    for _ in range(max_frames):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(observation.shape)
        rewards.append(reward)
        
        if terminated or truncated:
            break

    env.close()
    return rewards

def play_doom(num_envs, max_frames):
    # zero-padded rewards tensor
    rewards = np.zeros((num_envs, max_frames), dtype=np.float32)

    # Use ThreadPoolExecutor to run environments in parallel
    with ThreadPoolExecutor(max_workers=num_envs) as executor:
        futures = [executor.submit(run_single_episode, i, max_frames) for i in range(num_envs)]
        rewards_list = [future.result() for future in futures]

    for i, rewards_ in enumerate(rewards_list):
        rewards[i, :len(rewards_)] = rewards_
    
    return rewards

if __name__ == "__main__":
    num_envs = 8
    max_frames = 100
    
    # Roll out episodes in parallel
    rewards = play_doom(num_envs, max_frames)
    print("Rewards Tensor:", rewards.shape, rewards.sum(axis=1))
