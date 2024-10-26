import gymnasium
import os
import vizdoom as vzd
import cv2
from gymnasium.envs.registration import register
from vizdoom.gymnasium_wrapper import gymnasium_env_defns

# Register the custom scenario
# scenario_file = os.path.join(os.path.dirname(__file__), "scenarios", "oblige_custom.cfg")
scenario_file = os.path.join(os.path.dirname(__file__), "scenarios", "freedom_custom.cfg")
register(
    id="VizdoomCustom-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": scenario_file},
)

from dataclasses import dataclass

@dataclass
class VizDoomRewardFeatures:
    KILLCOUNT: int
    ITEMCOUNT: int
    SECRETCOUNT: int
    FRAGCOUNT: int
    DEATHCOUNT: int
    HITCOUNT: int
    HITS_TAKEN: int
    DAMAGECOUNT: int
    DAMAGE_TAKEN: int
    HEALTH: int
    ARMOR: int
    DEAD: int
    SELECTED_WEAPON_AMMO: int
    POSITION_X: int
    POSITION_Y: int
    POSITION_Z: int

    @classmethod
    def make_from_game(cls, game):
        return cls(
            KILLCOUNT=game.get_game_variable(vzd.GameVariable.KILLCOUNT),
            ITEMCOUNT=game.get_game_variable(vzd.GameVariable.ITEMCOUNT),
            SECRETCOUNT=game.get_game_variable(vzd.GameVariable.SECRETCOUNT),
            FRAGCOUNT=game.get_game_variable(vzd.GameVariable.FRAGCOUNT),
            DEATHCOUNT=game.get_game_variable(vzd.GameVariable.DEATHCOUNT),
            HITCOUNT=game.get_game_variable(vzd.GameVariable.HITCOUNT),
            HITS_TAKEN=game.get_game_variable(vzd.GameVariable.HITS_TAKEN),
            DAMAGECOUNT=game.get_game_variable(vzd.GameVariable.DAMAGECOUNT),
            DAMAGE_TAKEN=game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN),
            HEALTH=game.get_game_variable(vzd.GameVariable.HEALTH),
            ARMOR=game.get_game_variable(vzd.GameVariable.ARMOR),
            DEAD=game.get_game_variable(vzd.GameVariable.DEAD),
            SELECTED_WEAPON_AMMO=game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO),
            POSITION_X=game.get_game_variable(vzd.GameVariable.POSITION_X),
            POSITION_Y=game.get_game_variable(vzd.GameVariable.POSITION_Y),
            POSITION_Z=game.get_game_variable(vzd.GameVariable.POSITION_Z),
        )

    def get_deltas(self, other: "VizDoomRewardFeatures") -> "VizDoomRewardFeatures":
        # loop through all our fields and subtract the other fields from them
        field_names = self.__annotations__.keys()
        return VizDoomRewardFeatures(
            **{field: getattr(self, field) - getattr(other, field) for field in field_names}
        )


class VizDoomCustom:
    def __init__(self):
        self.env = gymnasium.make("VizdoomCustom-v0")
        self.game = self.env.env.env.game
        self._prev_reward_features = None
        self._current_reward_features = None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        observation, info = self.env.reset()
        self._prev_reward_features = self._get_reward_features()
        return observation, info

    def step(self, action):
        # Execute the action and observe the next state
        observation, _, terminated, truncated, info = self.env.step(action)
        self._current_reward_features = self._get_reward_features()

        # Calculate custom reward
        reward = self._get_reward()
        self._prev_reward_features = self._current_reward_features  # Update previous state

        return observation, reward, terminated, truncated, info

    def _get_reward_features(self) -> VizDoomRewardFeatures:
        return VizDoomRewardFeatures.make_from_game(self.game)

    def _get_reward(self):
        reward = 0

        if self._prev_reward_features is None:
            return reward

        # get deltas
        deltas = self._current_reward_features.get_deltas(self._prev_reward_features)

        reward += deltas.KILLCOUNT * 1000
        reward += deltas.ITEMCOUNT * 100
        reward += deltas.SECRETCOUNT * 500
        reward += deltas.HITCOUNT * 300
        reward += deltas.HEALTH * 10
        reward += deltas.ARMOR * 10
        reward += deltas.SELECTED_WEAPON_AMMO * 10

        return reward


# Run an example game loop
if __name__ == "__main__":
    agent = VizDoomCustom()

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
