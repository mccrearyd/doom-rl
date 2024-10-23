import gymnasium
from vizdoom import gymnasium_wrapper
env = gymnasium.make("VizdoomCorridor-v0")

observation, info = env.reset()

# let's do a cv2 window to show the screen
import cv2
cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
cv2.resizeWindow("screen", 640, 480)

for _ in range(1000):
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