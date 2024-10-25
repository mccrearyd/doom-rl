import gymnasium
from vizdoom import gymnasium_wrapper
from oblige import VizDoomOblige
import numpy as np
import pygame
import cv2

# Initialize environment
env = VizDoomOblige()
observation, info = env.reset()

# Initialize pygame
pygame.init()
WINDOW_WIDTH, WINDOW_HEIGHT = 800, 600
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Doom Environment")

# Action mappings
action_map = {
    "forward": 3,
    "backward": 4,
    "look_right": 2,
    "look_left": 1,
    "fire": 8
}

# Set up OpenCV window for additional screen output (if desired)
cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("screen", 640, 480)

# Main game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Key press handling
    keys = pygame.key.get_pressed()
    
    # Create action array based on key states (one-hot encoded)
    if keys[pygame.K_w]:  # Forward
        current_action = action_map["forward"]
    elif keys[pygame.K_s]:  # Backward
        current_action = action_map["backward"]
    elif keys[pygame.K_a] or keys[pygame.K_LEFT]:  # Look left
        current_action = action_map["look_left"]
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:  # Look right
        current_action = action_map["look_right"]
    elif keys[pygame.K_SPACE]:  # Fire
        current_action = action_map["fire"] 
    else:
        current_action = 0  # No action

    # Apply the action to the environment and update the state
    observation, reward, terminated, truncated, info = env.step(current_action)
    
    # Print reward for debugging
    print(f"Reward: {reward}")

    # Render the updated state (assuming observation["screen"] is the frame)
    img = np.array(observation["screen"]).astype(np.uint8)

    # Convert NumPy image to Pygame surface and blit it to the screen
    surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
    surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
    window.blit(surface, (0, 0))

    # Display the screen using OpenCV as an option
    cv2.imshow("screen", img)
    cv2.waitKey(1)

    # Update the display
    pygame.display.update()

    # Limit frame rate to 60 FPS
    pygame.time.Clock().tick(60)

    # Reset environment if done
    if terminated or truncated:
        observation, info = env.reset()

# Quit everything properly
env.close()
pygame.quit()
cv2.destroyAllWindows()
