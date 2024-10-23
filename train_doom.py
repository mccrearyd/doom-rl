from interactor import DoomInteractor

import torch
from torch import nn

import wandb



class Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # doom action space is Discrete(8) so we want
        # to output a distribution over 8 actions

        # observation shape is (240, 320, 3)
        # output should be a vector of 8 (our means)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),  # (32, 58, 78)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # (64, 28, 38)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # (64, 26, 36)
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # (64, 1, 1)
            nn.Flatten(),

            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=8),  # Final output shape is 8 (the action logits)
            torch.nn.Sigmoid(),
        )

        # temperature/std variable
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def get_distribution(self, means: torch.Tensor) -> torch.distributions.Categorical:
        dist = torch.distributions.Categorical(probs=means)
        return dist

    def forward(self, observations: torch.Tensor):
        # make float and reorder to (batch, channels, height, width) from (batch, height, width, channels)
        observations = observations.float().permute(0, 3, 1, 2)
        means = self.model(observations)
        dist = self.get_distribution(means)

        # actions = dist.sample()
        # print(dist)
        # print(actions.shape)
        # print(actions)
        # print(dist.log_prob(actions))
        # print(dist.entropy().mean())

        return dist


if __name__ == "__main__":
    USE_WANDB = True

    agent = Agent()

    NUM_VEPISODES = 8
    MAX_STEPS = 16
    NUM_ENVS = 4
    LR = 1e-4
    
    # if true one of the environments will be displayed in a cv2 window
    WATCH = False
    
    interactor = DoomInteractor(NUM_ENVS, watch=WATCH)

    # Reset all environments
    observations = interactor.env.reset()
    # print("Initial Observations:", observations.shape)

    cumulative_rewards = torch.zeros((NUM_ENVS,))
    log_probs = torch.zeros((NUM_ENVS,))
    entropies = torch.zeros((NUM_ENVS,))

    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    if USE_WANDB:
        wandb.init(project="doom-rl")
        wandb.watch(agent)

    for vepisode_i in range(NUM_VEPISODES):
        optimizer.zero_grad()

        # Example of stepping through the environments
        for step_i in range(MAX_STEPS):  # Step for 100 frames or episodes
            dist = agent.forward(observations.float())

            actions = dist.sample()
            entropy = dist.entropy()
            log_probs = dist.log_prob(actions)
            log_probs += log_probs
            entropies += entropy

            assert actions.shape == (NUM_ENVS,)
            assert log_probs.shape == (NUM_ENVS,)

            observations, rewards, dones = interactor.step()
            cumulative_rewards += rewards

        print("Cumulative Rewards:", cumulative_rewards)
        print("Log Probabilities:", log_probs)

        if USE_WANDB:
            wandb.log({
                "avg_vepisode_reward": cumulative_rewards.mean(),
                "avg_entropy": entropies.mean(),
                "avg_log_prob": log_probs.mean(),
                "vepisode": vepisode_i,
            })

        # loss is REINFORCE ie. -log_prob * reward
        loss = (-log_probs * cumulative_rewards).mean()
        print(loss.item())
        loss.backward()
        optimizer.step()

    # Close all environments
    interactor.env.close()
