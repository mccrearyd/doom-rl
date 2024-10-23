from interactor import DoomInteractor

import torch
from torch import nn



class Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # doom action space is Discrete(8) so we want
        # to output a distribution over 8 actions

        # observation shape is (240, 320, 3)
        # output should be a vector of 8 (our means)
        self.model = nn.Sequential(
            # 3 layers of conv + relu
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4),  # (32, 58, 78)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),  # (64, 28, 38)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # (64, 26, 36)
            nn.ReLU(),

            # Adaptive average pool to embedding size of 64
            nn.AdaptiveAvgPool2d((1, 1)),  # (64, 1, 1)

            # Flatten the tensor to feed into fully connected layers
            nn.Flatten(),

            # 2 layers of linear + relu
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=8),  # Final output shape is 8 (the action logits)
        )

        # temperature/std variable
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def get_distribution(self) -> torch.distributions.Normal:
        # 8 means with 1 std (self.temperature)
        pass


if __name__ == "__main__":
    agent = Agent()
    print(agent)
    exit()

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
