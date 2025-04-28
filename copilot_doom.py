import gymnasium
import numpy as np
import pygame
import cv2

from custom_doom import VizDoomCustom
from train_doom import Agent

import torch

# Initialize environment
env = VizDoomCustom(verbose=False)
obs, info = env.reset()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize agent
agent = Agent(obs_shape=obs["screen"].shape, num_discrete_actions=env.action_space.n, lr=0.001)
agent = agent.to(device)

# Initialize pygame
pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Copilot Doom")

# Action mappings
action_map = {
    "forward": 3,
    "backward": 4,
    "look_right": 2,
    "look_left": 1,
    "fire": 8,
    "use": 7,
}

clock = pygame.time.Clock()

# Stats
total_score = 0
no_ops_in_a_row = 0
PAUSE_NO_OPS = 60
FPS_LIMIT = 30
running = True

while running:
    # Pygame event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Agent predicts an action
    obs_tensor = torch.from_numpy(obs["screen"]).unsqueeze(0).to(device)  # (1, H, W, 3)
    actions, dist = agent.forward(obs_tensor.float(), greedy=False)

    agent_action = actions.item()  # (batch_size=1)

    # Human overrides
    keys = pygame.key.get_pressed()

    is_agent_action = False

    if keys[pygame.K_w]:
        current_action = action_map["forward"]
    elif keys[pygame.K_s]:
        current_action = action_map["backward"]
    elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
        current_action = action_map["look_left"]
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        current_action = action_map["look_right"]
    elif keys[pygame.K_SPACE]:
        current_action = action_map["fire"]
    elif keys[pygame.K_e]:
        current_action = action_map["use"]
    elif keys[pygame.K_ESCAPE]:
        obs, info = env.reset()
        continue
    elif keys[pygame.K_f]:
        # the user can force give agent actions by pressing "F"
        current_action = agent_action
        is_agent_action = True
    else:
        # no-op action
        current_action = 0

    # Track human inactivity
    if current_action == 0 or keys[pygame.K_f]:
        no_ops_in_a_row += 1
    else:
        no_ops_in_a_row = 0

    if no_ops_in_a_row > PAUSE_NO_OPS:
        current_action = agent_action
        is_agent_action = True

    # Actually step environment
    obs_next, reward, terminated, truncated, info = env.step(current_action)
    total_score += reward

    # Update agent (always training!)
    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
    done_tensor = torch.tensor([terminated or truncated], dtype=torch.float32, device=device)

    # using this IF would mean that only the player's actions can update the agent's weights
    if not is_agent_action:
        agent.update(torch.tensor(current_action), reward_tensor, done_tensor)

    # print update stats
    print(f"reward: {reward}, action: {current_action}")

    obs = obs_next

    # Pygame render
    img = np.array(obs["screen"]).astype(np.uint8)
    surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
    window.blit(surface, (0, 0))
    pygame.display.update()

    # FPS limit
    clock.tick(FPS_LIMIT)

    if terminated or truncated:
        print("Game Over!")
        print(f"Final Score: {total_score}")
        obs, info = env.reset()
        total_score = 0

pygame.quit()
