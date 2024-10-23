from interactor import DoomInteractor

import torch
from torch import nn

import wandb

class Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Doom action space is Discrete(8) so we want
        # to output a distribution over 8 actions

        # observation shape is (240, 320, 3)
        # output should be a vector of 8 (our means)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(in_features=8, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=8),  # Final output shape is 8 (the action logits)
            torch.nn.Sigmoid(),
        )

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_distribution(self, means: torch.Tensor) -> torch.distributions.Categorical:
        dist = torch.distributions.Categorical(probs=means)
        return dist

    def forward(self, observations: torch.Tensor):
        # make float and reorder to (batch, channels, height, width) from (batch, height, width, channels)
        observations = observations.float().permute(0, 3, 1, 2)
        means = self.model(observations)
        dist = self.get_distribution(means)
        return dist


if __name__ == "__main__":
    USE_WANDB = True  # Set to True to enable wandb logging

    agent = Agent()
    print(agent.num_params)

    VSTEPS = 32
    NUM_ENVS = 4
    LR = 1e-4
    
    WATCH = False
    
    interactor = DoomInteractor(NUM_ENVS, watch=WATCH)

    # Reset all environments
    observations = interactor.env.reset()

    cumulative_rewards = torch.zeros((NUM_ENVS,))
    cumulative_log_probs = torch.zeros((NUM_ENVS,))
    entropies = torch.zeros((NUM_ENVS,))

    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    if USE_WANDB:
        wandb.init(project="doom-rl", config={
            "num_parameters": agent.num_params,
        })
        wandb.watch(agent)

    # Example of stepping through the environments
    for step_i in range(VSTEPS):
        optimizer.zero_grad()

        dist = agent.forward(observations.float())
        actions = dist.sample()

        assert actions.shape == (NUM_ENVS,)

        entropy = dist.entropy()
        log_probs = dist.log_prob(actions)

        entropies += entropy
        cumulative_log_probs += log_probs

        observations, rewards, dones = interactor.step()
        cumulative_rewards += rewards

        # Reset cumulative rewards if done
        cumulative_rewards *= 1 - dones.float()

        # instantaneous loss
        loss = (-log_probs * cumulative_rewards).mean()

        loss.backward()
        optimizer.step()

        print(f"------------- {step_i} -------------")
        print(f"Loss:\t\t{loss.item():.4f}")
        print(f"Entropy:\t{entropies.mean().item():.4f}")
        print(f"Log Prob:\t{cumulative_log_probs.mean().item():.4f}")
        print(f"Reward:\t\t{cumulative_rewards.mean().item():.4f}")

        if USE_WANDB:
            wandb.log({
                "step": step_i,
                "avg_entropy": entropies.mean().item(),
                "avg_log_prob": cumulative_log_probs.mean().item(),
                "avg_reward": cumulative_rewards.mean().item(),
                "loss": loss.item(),
            })

    # Close all environments
    interactor.env.close()
