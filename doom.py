import gymnasium
from vizdoom import gymnasium_wrapper
env = gymnasium.make("VizdoomCorridor-v0")
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation["screen"].shape, reward)
    print(observation["gamevariables"].shape, reward)

    if terminated or truncated:
        observation, info = env.reset()


env.close()