import gymnasium
import os
import vizdoom as vzd
import cv2
from gymnasium.envs.registration import register
from vizdoom.gymnasium_wrapper import gymnasium_env_defns
import numpy as np
from copy import deepcopy

# Register the custom scenario
# scenario_file = os.path.join(os.path.dirname(__file__), "scenarios", "oblige_custom.cfg")
scenario_file = os.path.join(os.path.dirname(__file__), "scenarios", "freedom_custom.cfg")
register(
    id="VizdoomCustom-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": scenario_file},
)

from dataclasses import dataclass


def symlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))

@dataclass
class VizDoomRewardFeatures:
    # https://vizdoom.farama.org/api/python/enums/#vizdoom.GameVariable

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
    SELECTED_WEAPON: int
    POSITION_X: int
    POSITION_Y: int
    POSITION_Z: int

    TRAVELED_BOX: "TraveledBox"

    @classmethod
    def make_from_game(cls, game, traveled_box):
        # https://vizdoom.farama.org/api/python/enums/#vizdoom.GameVariable

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
            SELECTED_WEAPON=game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON),
            POSITION_X=game.get_game_variable(vzd.GameVariable.POSITION_X),
            POSITION_Y=game.get_game_variable(vzd.GameVariable.POSITION_Y),
            POSITION_Z=game.get_game_variable(vzd.GameVariable.POSITION_Z),
            TRAVELED_BOX=deepcopy(traveled_box),  # TODO: probably a better way to do this?
        )

    def get_deltas(self, other: "VizDoomRewardFeatures") -> "VizDoomRewardFeatures":
        # loop through all our fields and subtract the other fields from them
        field_names = self.__annotations__.keys()
        return VizDoomRewardFeatures(
            **{field: getattr(self, field) - getattr(other, field) for field in field_names}
        )
    
    def get_summary(self) -> str:
        # new line for every field
        summary = "-" * 20 + "\n"
        summary += "\n".join([f"{field}: {getattr(self, field)}" for field in self.__annotations__.keys()])
        return summary
    

@dataclass
class TraveledBox:
    """Tracks the volume of coverage in the map the player has traveled. Grows as the player moves around the map.
    Used for calculating deltas to view the amount of new map the player has explored.
    """

    min_x: int = None
    max_x: int = None
    min_y: int = None
    max_y: int = None
    min_z: int = None
    max_z: int = None

    @property
    def is_initialized(self):
        return self.min_x is not None

    def update(self, x, y, z):
        if not self.is_initialized:
            self.min_x = x
            self.max_x = x
            self.min_y = y
            self.max_y = y
            self.min_z = z
            self.max_z = z
            return

        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)
        self.min_z = min(self.min_z, z)
        self.max_z = max(self.max_z, z)

    def x_distance(self):
        if not self.is_initialized:
            return 0
        return self.max_x - self.min_x

    def y_distance(self):
        if not self.is_initialized:
            return 0
        return self.max_y - self.min_y
    
    def z_distance(self):
        if not self.is_initialized:
            return 0
        return self.max_z - self.min_z
    
    def average_distance(self):
        return (self.x_distance() + self.y_distance() + self.z_distance()) / 3
    
    def __sub__(self, other):
        # return the difference between average distances
        return abs(self.average_distance() - other.average_distance())

class VizDoomCustom:
    def __init__(self, verbose: bool = False):
        self.env = gymnasium.make("VizdoomCustom-v0")
        self.game = self.env.env.env.game
        self._prev_reward_features = None
        self._current_reward_features = None
        self.verbose = verbose
        self.traveled_box = TraveledBox()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self):
        observation, info = self.env.reset()
        self._prev_reward_features = self._get_reward_features()
        self._initial_reward_features = self._get_reward_features()
        self.traveled_box = TraveledBox()
        return observation, info

    def step(self, action):
        # Execute the action and observe the next state
        observation, _, terminated, truncated, info = self.env.step(action)
        self._current_reward_features = self._get_reward_features()

        # Calculate custom reward
        reward, deltas = self._get_reward()
        self._prev_reward_features = self._current_reward_features  # Update previous state

        info["deltas"] = deltas

        return observation, reward, terminated, truncated, info

    def _get_reward_features(self) -> VizDoomRewardFeatures:
        return VizDoomRewardFeatures.make_from_game(self.game, traveled_box=self.traveled_box)
    
    def verbose_print(self, *args):
        if self.verbose:
            print(*args)

    def _get_reward(self):
        # https://vizdoom.farama.org/api/python/enums/#vizdoom.GameVariable

        reward = 0

        if self._prev_reward_features is None:
            return reward
        
        # NOTE: must be before deltas are calculated
        # give some reward for the distance traveled from spawn
        # spawn_x, spawn_y, spawn_z = self._initial_reward_features.POSITION_X, self._initial_reward_features.POSITION_Y, self._initial_reward_features.POSITION_Z
        # last_x, last_y, last_z = self._prev_reward_features.POSITION_X, self._prev_reward_features.POSITION_Y, self._prev_reward_features.POSITION_Z
        current_x, current_y, current_z = self._current_reward_features.POSITION_X, self._current_reward_features.POSITION_Y, self._current_reward_features.POSITION_Z
        self.traveled_box.update(current_x, current_y, current_z)

        # get deltas
        deltas = self._current_reward_features.get_deltas(self._prev_reward_features)

        # map exploration reward
        # reward += deltas.TRAVELED_BOX
        reward += 1 / (self.traveled_box.average_distance() + 1)

        reward += deltas.KILLCOUNT * 1000
        reward += deltas.ITEMCOUNT * 10
        reward += deltas.SECRETCOUNT * 3000
        # reward += deltas.HITCOUNT * 100
        reward += deltas.DAMAGECOUNT * 10
        reward += deltas.HEALTH * 10
        reward += deltas.ARMOR * 10

        # 10x negative reward to DAMAGE_TAKEN
        reward -= deltas.DAMAGE_TAKEN * 10

        # NOTE: this is buggy - goes negative when picking up a better weapon
        # reward += deltas.SELECTED_WEAPON_AMMO * 10
        # we need to use SELECTED_WEAPON to see if this value changed. if
        # SELECTED_WEAPON is non-zero, then we know we picked up a new weapon, so
        # any ammo decrease should be ignored.
        if deltas.SELECTED_WEAPON != 0:
            # if we changed weapons, ignore ammo change reward, but give a nice reward
            reward += 1000
        else:
            # decrement reward for firing a weapon, unless we hit or killed an enemy
            landed_shot = deltas.KILLCOUNT != 0 or deltas.HITCOUNT != 0
            if not landed_shot:
                reward += deltas.SELECTED_WEAPON_AMMO * 30

        # decrement reward for taking damage (already covered in HEALTH and ARMOR)
        # reward -= deltas.DAMAGE_TAKEN * 10

        # decrement reward for dying
        reward -= deltas.DEAD * 100

        if reward != 0:
            self.verbose_print(deltas.get_summary())

        # return symlog(reward)
        return reward, deltas


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
