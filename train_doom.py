from interactor import DoomInteractor
from video import VideoTensorStorage

from custom_doom import VizDoomRewardFeatures
from typing import List

from gymnasium.spaces import Discrete

import torch
from torch import nn

import wandb

import os
import cv2
import numpy as np
import csv

import torch
import torch.nn as nn



def symlog_torch(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def _is_channel_first(shape: tuple) -> bool:
    assert 3 in shape, f"Image shape should have a 3 channel dimension, got {shape}"

    if len(shape) == 4:
        # check NCHW
        return shape[1] == 3
    elif len(shape) == 3:
        # check CHW
        return shape[0] == 3
    else:
        raise ValueError(f"Invalid shape: {shape}")


class Agent(torch.nn.Module):
    def __init__(self, obs_shape: tuple, num_discrete_actions: int):
        # NOTE: this agent was designed specifically for image observations and
        # a discrete action space.
        # should be a trivial change for new action spaces, but the observations
        # should still remain images (otherwise need to redesign other stuff like image
        # and video recordings).

        super().__init__()

        hidden_channels = 32
        embedding_size = 32

        self.hidden_channels = hidden_channels
        self.embedding_size = embedding_size

        if not _is_channel_first(obs_shape):
            obs_shape = (obs_shape[-1], *obs_shape[:-1])


        # 1. Observation Embedding: Convolutions + AdaptiveAvgPool + Flatten
        self.obs_embedding = nn.Sequential(
            torch.nn.LayerNorm(obs_shape),
            nn.Conv2d(in_channels=3, out_channels=hidden_channels, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1),
            nn.ReLU(),
            # nn.AdaptiveAvgPool2d((1, 1)),
            # just simple averaging across all channels
            # nn.AvgPool2d(kernel_size=3, stride=2),
        )

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_channels, out_features=embedding_size),
            nn.Sigmoid(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
            nn.Sigmoid(),
            # nn.Linear(in_features=embedding_size, out_features=embedding_size),
            # nn.Sigmoid(),
        )

        # Initialize hidden state to None; it will be dynamically set later
        self.hidden_state = None
        
        # 2. Embedding Blender: Combine the observation embedding and hidden state
        self.embedding_blender = nn.Sequential(
            nn.Linear(in_features=embedding_size * 2, out_features=embedding_size),
            nn.Sigmoid(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
            nn.Sigmoid(),
            # nn.Linear(in_features=embedding_size, out_features=embedding_size),
            # nn.Sigmoid(),
            # nn.Linear(in_features=embedding_size, out_features=embedding_size),
            # nn.Sigmoid(),
            # nn.Linear(in_features=embedding_size, out_features=embedding_size),
            # nn.Sigmoid(),
        )

        # 3. Action Head: Map blended embedding to action logits
        self.action_head = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=num_discrete_actions),
            nn.Sigmoid()
        )

    def reset(self, reset_mask: torch.Tensor):
        """Resets hidden states for the agent based on the reset mask."""
        batch_size = reset_mask.size(0)
        # Initialize hidden state to zeros where the reset mask is 1
        if self.hidden_state is None:
            self.hidden_state = torch.zeros(batch_size, self.embedding_size, device=reset_mask.device)

        # Reset hidden states for entries where reset_mask is True (done flags)
        self.hidden_state[reset_mask == 1] = 0

    def forward(self, observations: torch.Tensor):
        if not _is_channel_first(observations.shape):
            # need to make it NCHW
            observations = observations.float().permute(0, 3, 1, 2)
        
        # Get batch size to handle hidden state initialization if needed
        batch_size = observations.size(0)

        # Initialize hidden state if it's the first forward pass
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.embedding_size, device=observations.device)

        # 1. Get the observation embedding
        obs_embedding = self.obs_embedding(observations)
        # print(obs_embedding.shape, "obs emb shape after conv")
        # average across all channels
        obs_embedding = obs_embedding.mean(dim=(2, 3))
        # print(obs_embedding.shape, "obs emb shape after avg")
        obs_embedding = self.embedding_head(obs_embedding)

        # Detach the hidden state from the computation graph (to avoid gradient tracking)
        hidden_state = self.hidden_state.detach()

        # 2. Concatenate the observation embedding with the hidden state
        combined_embedding = torch.cat((obs_embedding, hidden_state), dim=1)

        # 3. Blend embeddings
        blended_embedding = self.embedding_blender(combined_embedding)

        # Update the hidden state for the next timestep without storing gradients
        # Ensure we do not modify inplace - create a new tensor
        self.hidden_state = blended_embedding.detach().clone()

        # 4. Compute action logits
        action_logits = self.action_head(blended_embedding)

        # 5. Return the action distribution
        dist = self.get_distribution(action_logits)

        # HACK: maybe we need a more general way to do this, but store
        # the previous action in the hidden state
        actions = dist.sample()
        self.hidden_state[:, -1] = actions

        return actions, dist

    def get_distribution(self, means: torch.Tensor) -> torch.distributions.Categorical:
        """Returns a categorical distribution over the action space."""
        dist = torch.distributions.Categorical(probs=means)
        return dist

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def timestamp_name():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



if __name__ == "__main__":
    USE_WANDB = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ENV_ID = "VizdoomCorridor-v0"
    # ENV_ID = "VizdoomDefendCenter-v0"
    # ENV_ID = "VizdoomDeathmatch-v0"
    ENV_ID = "VizdoomCustom-v0"

    VSTEPS = 10_000_000
    NUM_ENVS = 48
    GRID_SIZE = int(np.ceil(np.sqrt(NUM_ENVS)))  # Dynamically determine the grid size

    # LR = 1e-4  # works well for corridor
    LR = 1e-3

    TRAIN_ON_CUMULATIVE_REWARDS = True

    NORM_WITH_REWARD_COUNTER = True

    WATCH = False  # pop up display with live video frames

    # episode tracking (for video saving and replay)
    MAX_VIDEO_FRAMES = 1024  # will be clipped if a best episode is found to log to wandb
    MIN_EP_REWARD_SUM = 6000

    interactor = DoomInteractor(NUM_ENVS, watch=WATCH, env_id=ENV_ID)

    assert isinstance(interactor.single_action_space, Discrete), f"Expected Discrete action space, got {interactor.single_action_space}"

    agent = Agent(obs_shape=interactor.env.obs_shape, num_discrete_actions=interactor.single_action_space.n)
    
    # remove the 3 from the shape
    _obs_shape = interactor.env.obs_shape
    _obs_shape = tuple([x for x in _obs_shape if x != 3])
    assert len(_obs_shape) == 2, "Observation shape should be 2D after removing the channel dimension"
    FRAME_HEIGHT, FRAME_WIDTH = _obs_shape

    agent = agent.to(device)
    print(agent.num_params)

    # Reset all environments
    observations = interactor.reset()

    cumulative_rewards = torch.zeros((NUM_ENVS,))
    cumulative_rewards_no_reset = torch.zeros((NUM_ENVS,))
    step_counters = torch.zeros((NUM_ENVS,), dtype=torch.float32)

    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    best_episode_cumulative_reward = -float("inf")
    best_episode_env = None
    best_episode = None

    # Initialize wandb project
    if USE_WANDB:
        wandb.init(project=f"doom-rl-{ENV_ID}", config={
            "num_parameters": agent.num_params,
            "v_steps": VSTEPS,
            "num_envs": NUM_ENVS,
            "lr": LR,
            "norm_with_reward_counter": NORM_WITH_REWARD_COUNTER,
            "obs_shape": interactor.env.obs_shape,
            "num_discrete_actions": interactor.single_action_space.n,
            "env_id": ENV_ID,
            "agent": agent,
        })
        wandb.watch(agent)

    run_name = wandb.run.name if USE_WANDB else timestamp_name()
    video_path = os.path.join("trajectory_videos", f"{ENV_ID}/{run_name}")
    video_storage = VideoTensorStorage(
        folder=video_path,
        max_video_frames=MAX_VIDEO_FRAMES, grid_size=GRID_SIZE,
        frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH, num_envs=NUM_ENVS
    )

    num_kills_all_time = 0
    damage_taken_all_time = 0
    secrets_found_all_time = 0
    death_count_all_time = 0
    num_resets_all_time = 0

    # Example of stepping through the environments
    for step_i in range(VSTEPS):
        optimizer.zero_grad()

        actions, dist = agent.forward(observations.float().to(device))

        assert actions.shape == (NUM_ENVS,)

        entropy = dist.entropy()
        log_probs = dist.log_prob(actions)

        observations, rewards, dones, infos = interactor.step(actions.cpu().numpy())

        cumulative_rewards += rewards
        cumulative_rewards_no_reset += rewards

        # Update the video storage with the new frame and episode tracking
        video_storage.update_and_save_frame(observations, dones)

        episodic_rewards = []
        for i, done in enumerate(dones):
            if done:
                episodic_rewards.append(cumulative_rewards[i].item())

                # TODO: criteria for best episode maybe should be most kills
                if cumulative_rewards[i].item() > best_episode_cumulative_reward:
                    best_episode_cumulative_reward = cumulative_rewards[i].item()
                    best_episode_env = i  # Track which environment achieved the best reward
                    best_episode = int(video_storage.episode_counters[i].item())  # Track the episode number

        episodic_rewards = torch.tensor(episodic_rewards)

        # Reset cumulative rewards if done
        cumulative_rewards *= 1 - dones.float()

        # count the number of steps taken (reset if done)
        step_counters += 1
        step_counters *= 1 - dones.float()

        # call agent.reset with done flags for hidden state resetting
        agent.reset(dones)

        logging_cumulative_rewards = cumulative_rewards.clone()

        if NORM_WITH_REWARD_COUNTER:
            cumulative_rewards /= step_counters + 1

        if TRAIN_ON_CUMULATIVE_REWARDS:
            # cumulative rewards
            scores = cumulative_rewards
        else:
            # instantaneous rewards
            scores = rewards

        norm_scores = (scores - scores.mean()) / (scores.std() + 1e-8)

        # specifically symlog after normalizing scores
        norm_scores = symlog_torch(norm_scores)
        loss = (-log_probs * norm_scores.to(device)).mean()

        loss.backward()
        optimizer.step()

        print(f"------------- {step_i} -------------")
        print(f"Loss:\t\t{loss.item():.4f}")
        print(f"Entropy:\t{entropy.mean().item():.4f}")
        print(f"Log Prob:\t{log_probs.mean().item():.4f}")
        print(f"Reward:\t\t{rewards.mean().item():.4f}")

        # If we have a new best episode, log the video to wandb
        if best_episode_cumulative_reward > MIN_EP_REWARD_SUM and USE_WANDB:
            if best_episode_env is not None and best_episode is not None:
                print(f"New best episode found for environment {best_episode_env}, episode {best_episode}!")

                # Extract the video slice for the best episode and environment
                video_slice_tensor = video_storage.get_video_slice(env_i=best_episode_env, episode=best_episode - 1)

                # Log the video slice to wandb
                if video_slice_tensor.size(0) > 0:  # Ensure the tensor has frames
                    video_np = video_slice_tensor.cpu().numpy()

                    highlight_path = os.path.join(video_path, "highlights")
                    os.makedirs(highlight_path, exist_ok=True)
                    highlight_file_path = os.path.join(highlight_path, f"env_{best_episode_env}-ep_{best_episode}.mp4")

                    height, width = video_np.shape[2], video_np.shape[3]
                    out = cv2.VideoWriter(highlight_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (video_storage.frame_width, video_storage.frame_height))

                    # write each frame (it expects shape to be HWC)
                    for frame in video_np:
                        out.write(frame.transpose(1, 2, 0))

                    out.release()

                    # TODO: fix wandb video logging
                    # wandb_video = wandb.Video(highlight_file_path, format="mp4")
                    # wandb.log({
                    #     "best_episode_video": wandb_video,
                    # }, commit=False)

                # Reset the best episode tracking after logging
                best_episode_env = None
                best_episode = None

        # Log wandb metrics
        if USE_WANDB:
            for info in infos:
                if info.get("was_reset", False):
                    num_resets_all_time += 1
                    continue

                if "deltas" not in info:
                    continue

                deltas = info["deltas"]
                num_kills_all_time += deltas.KILLCOUNT
                damage_taken_all_time += deltas.DAMAGE_TAKEN
                secrets_found_all_time += deltas.SECRETCOUNT
                death_count_all_time += deltas.DEATHCOUNT

            data = {
                "step": step_i,
                "avg_entropy": entropy.mean().item(),
                "avg_log_prob": log_probs.mean().item(),
                "num_done": dones.sum().item(),
                "loss": loss.item(),
                "scores/num_resets_all_time": num_resets_all_time,
                "scores/num_kills_all_time": num_kills_all_time,
                "scores/damage_taken_all_time": damage_taken_all_time,
                "scores/secrets_found_all_time": secrets_found_all_time,
                "scores/death_count_all_time": death_count_all_time,
                "rewards/best_episodic_reward": best_episode_cumulative_reward,
                "rewards/avg_instantaneous_reward": rewards.mean().item(),
                "rewards/avg_cumulative_reward": logging_cumulative_rewards.mean().item(),
                "rewards/avg_cumulative_reward_no_reset": cumulative_rewards_no_reset.mean().item(),
            }

            if len(episodic_rewards) > 0:
                data["episodic_rewards"] = episodic_rewards.mean()

            wandb.log(data)

    video_storage.close()  # Close video storage after the loop ends

    # Close all environments
    interactor.env.close()
