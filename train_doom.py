from interactor import DoomInteractor

import torch
from torch import nn

import wandb

import os
import cv2
import numpy as np
import csv

import torch
import torch.nn as nn

class Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Doom action space is Discrete(8), so we want to output a distribution over 8 actions
        hidden_channels = 64
        embedding_size = 64

        self.hidden_channels = hidden_channels
        self.embedding_size = embedding_size

        # output should be a vector of 8 (our means)

        # obs_shape = (3, 180, 320)  # oblige
        obs_shape = (3, 240, 320)
        
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
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
            nn.Sigmoid(),
        )

        # Initialize hidden state to None; it will be dynamically set later
        self.hidden_state = None
        
        # 2. Embedding Blender: Combine the observation embedding and hidden state
        self.embedding_blender = nn.Sequential(
            nn.Linear(in_features=embedding_size * 2, out_features=embedding_size),
            nn.Sigmoid(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
            nn.Sigmoid(),
            nn.Linear(in_features=embedding_size, out_features=embedding_size),
            nn.Sigmoid(),
        )

        # 3. Action Head: Map blended embedding to action logits
        self.action_head = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=8),
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
        # Reorder observations to (batch, channels, height, width) from (batch, height, width, channels)
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


if __name__ == "__main__":
    USE_WANDB = False  # Disable wandb logging for now

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent()
    agent = agent.to(device)
    print(agent.num_params)

    VSTEPS = 10_000_000
    NUM_ENVS = 48
    GRID_SIZE = int(np.ceil(np.sqrt(NUM_ENVS)))  # Dynamically determine the grid size
    LR = 1e-4

    NORM_WITH_REWARD_COUNTER = False

    WATCH = True
    MAX_VIDEO_FRAMES = 100
    FRAME_HEIGHT = 240
    FRAME_WIDTH = 320

    interactor = DoomInteractor(NUM_ENVS, watch=WATCH)

    # Create directory for storing videos and CSVs
    os.makedirs('trajectory_videos', exist_ok=True)

    # Reset all environments
    observations = interactor.reset()

    cumulative_rewards = torch.zeros((NUM_ENVS,))
    step_counters = torch.zeros((NUM_ENVS,), dtype=torch.float32)
    episode_counters = torch.zeros((NUM_ENVS,), dtype=torch.int32)  # Track episode numbers for each environment

    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    # Video capture settings
    video_file_count = 0
    frame_count = 0
    video_writer = None
    episode_tracker = []  # This will track episode numbers for each frame

    def open_video_writer():
        global video_writer, video_file_count
        video_file_count += 1
        video_path = os.path.join(f"trajectory_videos", f"frames_{video_file_count}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (FRAME_WIDTH * GRID_SIZE, FRAME_HEIGHT * GRID_SIZE))

    def close_video_writer():
        global video_writer
        if video_writer:
            video_writer.release()
            video_writer = None

    def save_episode_csv():
        # Save the episode tracking information to a CSV file
        csv_path = os.path.join(f"trajectory_videos", f"episodes_{video_file_count}.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(episode_tracker)

    open_video_writer()  # Start the first video file

    best_episode_cumulative_reward = -float("inf")

    # Example of stepping through the environments
    for step_i in range(VSTEPS):
        optimizer.zero_grad()

        actions, dist = agent.forward(observations.float().to(device))

        assert actions.shape == (NUM_ENVS,)

        entropy = dist.entropy()
        log_probs = dist.log_prob(actions)

        observations, rewards, dones = interactor.step(actions.cpu().numpy())
        cumulative_rewards += rewards

        # Create the tiled video frame grid
        frames = observations.cpu().numpy()  # Assuming observations are the frames in shape (NUM_ENVS, H, W, C)
        grid_frame = np.zeros((FRAME_HEIGHT * GRID_SIZE, FRAME_WIDTH * GRID_SIZE, 3), dtype=np.uint8)

        for i in range(NUM_ENVS):
            row = i // GRID_SIZE
            col = i % GRID_SIZE
            grid_frame[
                row * FRAME_HEIGHT:(row + 1) * FRAME_HEIGHT,
                col * FRAME_WIDTH:(col + 1) * FRAME_WIDTH
            ] = frames[i]

        # Write the tiled frame to the video
        video_writer.write(cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR))

        # Update the episode tracker for each frame
        episode_tracker.append(episode_counters.clone().tolist())

        # Manage video file splitting
        frame_count += 1
        if frame_count >= MAX_VIDEO_FRAMES:
            close_video_writer()
            save_episode_csv()  # Save the CSV file for the current video segment
            open_video_writer()
            frame_count = 0
            episode_tracker = []  # Reset tracker for the next chunk of video

        # any time the environment is done, before resetting the cumulative rewards, let's log it to
        # episodic_rewards
        episodic_rewards = []
        for i, done in enumerate(dones):
            if done:
                episodic_rewards.append(cumulative_rewards[i].item())
                episode_counters[i] += 1  # Increment episode count for this environment

                if cumulative_rewards[i].item() > best_episode_cumulative_reward:
                    best_episode_cumulative_reward = cumulative_rewards[i].item()

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

        norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        loss = (-log_probs * norm_rewards.to(device)).mean()

        loss.backward()
        optimizer.step()

        print(f"------------- {step_i} -------------")
        print(f"Loss:\t\t{loss.item():.4f}")
        print(f"Entropy:\t{entropy.mean().item():.4f}")
        print(f"Log Prob:\t{log_probs.mean().item():.4f}")
        print(f"Reward:\t\t{rewards.mean().item():.4f}")

    close_video_writer()  # Close the last video file when done
    save_episode_csv()  # Save the CSV for the last video segment

    # Close all environments
    interactor.env.close()
