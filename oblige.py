import gymnasium
import os
import vizdoom as vzd
import cv2
from gymnasium.envs.registration import register
from vizdoom.gymnasium_wrapper import gymnasium_env_defns

# Register the custom scenario
scenario_file = os.path.join(os.path.dirname(__file__), "scenarios", "oblige_custom.cfg")
register(
    id="VizdoomOblige-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": scenario_file},
)

class VizDoomOblige:
    def __init__(self):
        self.env = gymnasium.make("VizdoomOblige-v0")
        self.game = self.env.env.env.game
        self.prev_state = None

    def reset(self):
        observation, info = self.env.reset()
        self.prev_state = self._get_state()
        return observation, info

    def step(self, action):
        # Execute the action and observe the next state
        observation, _, terminated, truncated, info = self.env.step(action)
        next_state = self._get_state()

        # Calculate custom reward
        reward = self._get_reward(self.prev_state, next_state)
        self.prev_state = next_state  # Update previous state

        return observation, reward, terminated, truncated, info

    def _get_state(self):
        return (
            self.game.get_game_variable(vzd.GameVariable.KILLCOUNT),
            self.game.get_game_variable(vzd.GameVariable.ITEMCOUNT),
            self.game.get_game_variable(vzd.GameVariable.SECRETCOUNT),
            self.game.get_game_variable(vzd.GameVariable.FRAGCOUNT),
            self.game.get_game_variable(vzd.GameVariable.DEATHCOUNT),
            self.game.get_game_variable(vzd.GameVariable.HITCOUNT),
            self.game.get_game_variable(vzd.GameVariable.HITS_TAKEN),
            self.game.get_game_variable(vzd.GameVariable.DAMAGECOUNT),
            self.game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN),
            self.game.get_game_variable(vzd.GameVariable.HEALTH),
            self.game.get_game_variable(vzd.GameVariable.ARMOR),
            self.game.get_game_variable(vzd.GameVariable.DEAD),
            self.game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO),
            self.game.get_game_variable(vzd.GameVariable.POSITION_X),
            self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
            self.game.get_game_variable(vzd.GameVariable.POSITION_Z),
        )

    def _get_reward(self, prev_state, next_state):
        reward = 0
        # Reward calculations based on state changes
        reward += (next_state[5] - prev_state[5]) * 300  # Enemy hit
        reward += (next_state[0] - prev_state[0]) * 1000  # Enemy kill
        reward += (next_state[1] - prev_state[1]) * 100  # Item pick up
        reward += (next_state[2] - prev_state[2]) * 500  # Secret found
        reward += (next_state[9] - prev_state[9]) * 10  # Health delta
        reward += (next_state[10] - prev_state[10]) * 10  # Armor delta
        reward += (next_state[12] - prev_state[12]) * 10  # Ammo delta

        return reward


# Run an example game loop
if __name__ == "__main__":
    agent = VizDoomOblige()

    # Reset environment
    observation, info = agent.reset()

    # Initialize CV2 window
    cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("screen", 640, 480)

    # Run loop
    for _ in range(1000000):
        action = agent.env.action_space.sample()
        observation, reward, terminated, truncated, info = agent.step(action)

        # Display the reward and the screen
        print(reward)
        cv2.imshow("screen", observation["screen"])
        cv2.waitKey(1)

        if terminated or truncated:
            observation, info = agent.reset()

    agent.env.close()
