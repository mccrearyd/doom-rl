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
    USE_WANDB = False  # Set to True to enable wandb logging

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent()
    agent = agent.to(device)
    print(agent.num_params)

    VSTEPS = 100_000
    NUM_ENVS = 8
    LR = 1e-5

    NORM_WITH_REWARD_COUNTER = False
    
    WATCH = False
    
    interactor = DoomInteractor(NUM_ENVS, watch=WATCH)

    # Reset all environments
    observations = interactor.reset()

    cumulative_rewards = torch.zeros((NUM_ENVS,))
    step_counters = torch.zeros((NUM_ENVS,), dtype=torch.float32)

    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)

    if USE_WANDB:
        wandb.init(project="doom-rl", config={
            "num_parameters": agent.num_params,
        })
        wandb.watch(agent)

    # Example of stepping through the environments
    for step_i in range(VSTEPS):
        optimizer.zero_grad()

        dist = agent.forward(observations.float().to(device))
        actions = dist.sample()

        assert actions.shape == (NUM_ENVS,)

        entropy = dist.entropy()
        log_probs = dist.log_prob(actions)

        observations, rewards, dones = interactor.step(actions.cpu().numpy())
        cumulative_rewards += rewards

        # any time the environment is done, before resetting the cumulative rewards, let's log it to
        # episodic_rewards
        episodic_rewards = []
        for i, done in enumerate(dones):
            if done:
                episodic_rewards.append(cumulative_rewards[i].item())
        episodic_rewards = torch.tensor(episodic_rewards)

        # Reset cumulative rewards if done
        cumulative_rewards *= 1 - dones.float()

        # count the number of steps taken (reset if done)
        step_counters += 1
        step_counters *= 1 - dones.float()

        # print(f"Step Counters: {step_counters}")

        if NORM_WITH_REWARD_COUNTER:
            # average cumulative rewards over the number of steps taken
            # cumulative_rewards = cumulative_rewards / (step_counters + 1)
            cumulative_rewards /= step_counters + 1

        # instantaneous loss
        # norm_rewards = (rewards - cumulative_rewards.mean()) / (cumulative_rewards.std() + 1e-8)
        norm_rewards = (cumulative_rewards - cumulative_rewards.mean()) / (cumulative_rewards.std() + 1e-8)

        loss = (-log_probs * norm_rewards.to(device)).mean()

        loss.backward()
        optimizer.step()

        print(f"------------- {step_i} -------------")
        print(f"Loss:\t\t{loss.item():.4f}")
        print(f"Entropy:\t{entropy.mean().item():.4f}")
        print(f"Log Prob:\t{log_probs.mean().item():.4f}")
        print(f"Reward:\t\t{cumulative_rewards.mean().item():.4f}")

        if USE_WANDB:
            data = {
                "step": step_i,
                "avg_entropy": entropy.mean().item(),
                "avg_log_prob": log_probs.mean().item(),
                "avg_reward": cumulative_rewards.mean().item(),
                "num_done": dones.sum().item(),
                "loss": loss.item(),
            }

            if len(episodic_rewards) > 0:
                data["episodic_rewards"] = episodic_rewards.mean()

            wandb.log(data)

    # Close all environments
    interactor.env.close()
