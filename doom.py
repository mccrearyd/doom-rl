import gymnasium
from vizdoom import gymnasium_wrapper
env = gymnasium.make("VizdoomCorridor-v0")

observation, info = env.reset()

# let's do a cv2 window to show the screen
import cv2
cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("screen", 640, 480)

for _ in range(1000):
    # action = 0  # does nothing
    # action = 1  # turns mouse to the right
    # action = 2  # turns mouse to the left
    # action = 3  # walks backwards?
    # action = 4  # walks forwards
    # action = 5  # fires
    # action = 6  # ??? back right ???
    # action = 7  # ??? back left ???
    # action = 0

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)
    print(env.action_space)
    print(action)
    # print(observation["screen"].shape, reward)
    # print(observation["screen"].dtype)
    # print(observation["gamevariables"].shape, reward)

    # show the screen
    cv2.imshow("screen", observation["screen"])
    cv2.waitKey(1)

    if terminated or truncated:
        observation, info = env.reset()


env.close()