from interactor import DoomInteractor
from video import VideoTensorStorage

from custom_doom import VizDoomRewardFeatures
from typing import List

from argparse import ArgumentParser

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
    

def multi_sample_argmax(dist: torch.distributions.Distribution, k: int = 3):
    # Sample 'k' times for each distribution in the batch
    actions = dist.sample((k,))
    
    # Calculate log probabilities for each sample
    log_probs = dist.log_prob(actions)
    
    # Find the index of the maximum log probability for each element in the batch
    max_indices = torch.argmax(log_probs, dim=0)
    
    # Gather the actions corresponding to the maximum log probabilities
    best_actions = actions.gather(0, max_indices.unsqueeze(0)).squeeze(0)

    return best_actions


class Agent(nn.Module):
    def __init__(self, obs_shape: tuple, num_discrete_actions: int, lr: float = 5e-4):
        super().__init__()

        hidden_channels = 16
        embedding_size = 32

        if not _is_channel_first(obs_shape):
            obs_shape = (obs_shape[-1], *obs_shape[:-1])

        self.embedding_size = embedding_size
        self.obs_embedding = nn.Sequential(
            nn.LayerNorm(obs_shape),
            nn.Conv2d(3, hidden_channels, 7, 3),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, 2),
            nn.ReLU(),
        )

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, embedding_size),
            nn.Sigmoid(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid(),
        )

        self.embedding_blender = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.Sigmoid(),
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid(),
        )

        self.action_head = nn.Sequential(
            nn.Linear(embedding_size, num_discrete_actions),
            nn.Sigmoid()
        )

        self.hidden_state = None
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Step tracking
        self.step_counters = None
        self.train_on_cumulative_rewards = False
        self.norm_with_step_counter = False
        self.batch_norm_rewards = False

        # Save latest log_probs for loss computation
        self.latest_dist = None

    def reset(self, reset_mask: torch.Tensor):
        """Reset hidden states and step counters when episode is done."""
        batch_size = reset_mask.size(0)

        if self.hidden_state is None:
            self.hidden_state = torch.zeros(batch_size, self.embedding_size, device=reset_mask.device)

        self.hidden_state[reset_mask == 1] = 0

        if self.step_counters is None:
            self.step_counters = torch.zeros((batch_size,), dtype=torch.float32, device=reset_mask.device)

        self.step_counters[reset_mask == 1] = 0

    def forward(self, observations: torch.Tensor, greedy: bool = False):
        if not _is_channel_first(observations.shape):
            observations = observations.float().permute(0, 3, 1, 2)

        batch_size = observations.size(0)

        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.hidden_state = torch.zeros(batch_size, self.embedding_size, device=observations.device)

        obs_embedding = self.obs_embedding(observations)
        obs_embedding = obs_embedding.mean(dim=(2, 3))
        obs_embedding = self.embedding_head(obs_embedding)

        hidden_state = self.hidden_state.detach()
        combined_embedding = torch.cat((obs_embedding, hidden_state), dim=1)
        blended_embedding = self.embedding_blender(combined_embedding)

        self.hidden_state = blended_embedding.detach().clone()

        action_logits = self.action_head(blended_embedding)
        dist = self.get_distribution(action_logits)

        if greedy:
            # choose the highest probability (mean) action
            actions = dist.probs.argmax(dim=1)
        else:
            # actions = multi_sample_argmax(dist, k=3)
            actions = dist.sample()

        # self.hidden_state[:, -1] = actions
        # self.latest_log_probs = dist.log_prob(actions)
        self.latest_dist = dist

        return actions, dist

    def get_distribution(self, means: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(probs=means)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def update(self, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor):
        """Handles loss calculation, backpropagation and optimizer step."""

        dist = self.latest_dist

        # Update step counters
        if self.step_counters is None:
            self.step_counters = torch.zeros_like(rewards, dtype=torch.float32)
        self.step_counters += 1
        self.step_counters *= (1 - dones.float())

        # Reset hidden state and counters where needed
        self.reset(dones)

        scores = rewards.clone()

        if self.train_on_cumulative_rewards:
            if self.norm_with_step_counter:
                scores /= (self.step_counters + 1)
        if self.batch_norm_rewards:
            scores = (scores - scores.mean()) / (scores.std() + 1e-8)

        log_probs = dist.log_prob(actions)
        loss = (-log_probs * scores).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "entropy": self.latest_dist.entropy().mean().item(),
            "log_prob": log_probs.mean().item(),
            "reward": rewards.mean().item()
        }


def timestamp_name():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def mini_cli():
    parser = ArgumentParser()
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--watch", action="store_true", default=False)
    parser.add_argument("--save", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = mini_cli()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ENV_ID = "VizdoomCorridor-v0"
    # ENV_ID = "VizdoomDefendCenter-v0"
    # ENV_ID = "VizdoomDeathmatch-v0"
    ENV_ID = "VizdoomCustom-v0"

    VSTEPS = 10_000_000
    NUM_ENVS = 8
    GRID_SIZE = int(np.ceil(np.sqrt(NUM_ENVS)))  # Dynamically determine the grid size

    # LR = 1e-4  # works well for corridor
    LR = 5e-4

    TRAIN_ON_CUMULATIVE_REWARDS = False
    NORM_WITH_REWARD_COUNTER = False

    # episode tracking (for video saving and replay)
    MAX_VIDEO_FRAMES = 1024  # will be clipped if a best episode is found to log to wandb
    MIN_EP_REWARD_SUM = 6000

    run_name = timestamp_name()  # TODO: bring back wandb run names
    # run_name = wandb.run.name if args.use_wandb else timestamp_name()
    trajectory_videos_path = os.path.join("trajectory_videos", ENV_ID)
    video_path = os.path.join(trajectory_videos_path, run_name)

    if args.save:
        watch_path = os.path.join(video_path, "watch.mp4")
    else:
        watch_path = None

    interactor = DoomInteractor(NUM_ENVS, watch=args.watch, watch_video_path=watch_path, env_id=ENV_ID)

    assert isinstance(interactor.single_action_space, Discrete), f"Expected Discrete action space, got {interactor.single_action_space}"
    
    # remove the 3 from the shape
    _obs_shape = interactor.env.obs_shape
    _obs_shape = tuple([x for x in _obs_shape if x != 3])
    assert len(_obs_shape) == 2, "Observation shape should be 2D after removing the channel dimension"
    FRAME_HEIGHT, FRAME_WIDTH = _obs_shape

    video_storage = VideoTensorStorage(
        folder=video_path,
        max_video_frames=MAX_VIDEO_FRAMES, grid_size=GRID_SIZE,
        frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH, num_envs=NUM_ENVS
    )

    agent = Agent(
        obs_shape=interactor.env.obs_shape,
        num_discrete_actions=interactor.single_action_space.n,
        lr=LR
    ).to(device)

    print(agent.num_params)

    observations = interactor.reset()
    cumulative_rewards_no_reset = torch.zeros((NUM_ENVS,))
    best_episode_cumulative_reward = -float("inf")
    best_episode_env = None
    best_episode = None

    try:
        for step_i in range(VSTEPS):
            actions, dist = agent.forward(observations.float().to(device))

            assert actions.shape == (NUM_ENVS,)

            interactor.watch_index = 0 if best_episode_env is None else best_episode_env

            observations, rewards, dones, infos = interactor.step(actions.cpu().numpy())

            cumulative_rewards_no_reset += rewards

            video_storage.update_and_save_frame(observations, dones)

            episodic_rewards = []
            for i in range(NUM_ENVS):
                if dones[i]:
                    episodic_rewards.append(interactor.current_episode_cumulative_rewards[i].item())
                if interactor.current_episode_cumulative_rewards[i].item() > best_episode_cumulative_reward:
                    best_episode_cumulative_reward = interactor.current_episode_cumulative_rewards[i].item()
                    best_episode_env = i
                    best_episode = int(video_storage.episode_counters[i].item())

            stats = agent.update(actions, rewards.to(device), dones.to(device))

            print(f"------------- {step_i} -------------")
            print(f"Loss:\t\t{stats['loss']:.4f}")
            print(f"Entropy:\t{stats['entropy']:.4f}")
            print(f"Log Prob:\t{stats['log_prob']:.4f}")
            print(f"Reward:\t\t{stats['reward']:.4f}")

            if args.use_wandb:
                wandb.log({
                    "step": step_i,
                    "loss": stats["loss"],
                    "entropy": stats["entropy"],
                    "log_prob": stats["log_prob"],
                    "reward": stats["reward"],
                })

    except KeyboardInterrupt as e:
        print("Interrupted by user, finalizing data...")
        video_storage.close()
        interactor.env.close()
        raise e

