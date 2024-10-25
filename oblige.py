import os
import sys
import time
import json
import numpy as np
import cv2
import gc
from moviepy.editor import ImageSequenceClip
import gymnasium as gym
from gymnasium.envs.registration import register
import functools
# import pufferlib
import vizdoom as vzd

scenario_file = os.path.join(os.path.dirname(__file__), "scenarios", "oblige_custom.cfg")
register(
    id="VizdoomOblige-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file": scenario_file},
)

# def env_creator(name='VizdoomOblige-v0'):
#     return functools.partial(make, name)

# def make(name, framestack=1, render_mode='rgb_array'):
#     from vizdoom import gymnasium_wrapper
#     # Suppress unnecessary output during environment creation
#     with pufferlib.utils.Suppress():
#         env = gym.make(name, render_mode=render_mode, large_screen=True)
    
#     # from stable_baselines3.common.atari_wrappers import (
#     #     ClipRewardEnv,
#     #     EpisodicLifeEnv,
#     #     FireResetEnv,
#     #     MaxAndSkipEnv,
#     #     NoopResetEnv,
#     # )


    

    # env = DoomWrapper(env)
    # # env = MaxAndSkipEnv(env, skip=2)
    # return pufferlib.emulation.GymnasiumPufferEnv(env=env)

# def make_eps_env(name='VizdoomOblige-v0'):
#     with pufferlib.utils.Suppress():
#         env = gym.make(name, render_mode='rgb_array')
#     env = DoomWrapperEps(env)
#     return env#pufferlib.emulation.GymnasiumPufferEnv(env=)

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray[..., np.newaxis].astype(np.uint8)

def resize_frame(frame, new_scale):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w*new_scale), int(h*new_scale)))

def flatten_obs(frame, gamemap, actions):
    # Pre-allocate the output array to avoid multiple concatenations
    total_size = frame.size + gamemap.size + actions.size
    flattened = np.empty(total_size, dtype=np.uint8)
    
    # Use direct indexing instead of concatenate
    pos = 0
    flattened[pos:pos + frame.size] = frame.ravel()
    pos += frame.size
    flattened[pos:pos + gamemap.size] = gamemap.ravel()
    pos += gamemap.size
    flattened[pos:] = actions
    
    return flattened

OBS_SPACE = (90*160*3*4) + (90*160*3*4) + 32

def symlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))

class DoomWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env.unwrapped)
        if env.observation_space['screen'].shape[0] != 180:
            raise ValueError(
                'Wrong screen resolution. Doom does not provide '
                'a way to change this. You must edit scenarios/<env_name>.cfg.'
            )

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(OBS_SPACE,), dtype=np.uint8
        )
        
        self.video_folder = '/home/ubuntu/vizdoom/episodes'
        os.makedirs(self.video_folder, exist_ok=True)
        
        # Use deque with maxlen for better memory management
        from collections import deque
        self.frame_buffer = deque(maxlen=16)  # Limit frame buffer size
        self.game_map_buffer = deque(maxlen=16)  # Limit game map buffer size
        self.action_buffer = deque(maxlen=1000)  # Limit action buffer size
        self.reward_buffer = deque(maxlen=1000)  # Limit reward buffer size
        
        self.episode_count = 0
        self.exploration_tracker = ExplorationTracker(75)
        self.past_action = None
        self.repeat_count = 0
        self.count_by_class = {}

    def get_padded_last_32_actions(self):
        past_32 = np.zeros(32, dtype=np.uint8)
        if self.action_buffer:
            actions = list(self.action_buffer)[-32:]
            past_32[-len(actions):] = np.array(actions) + 1
        return past_32

    def get_gamemap(self):
        if (self.state is None) or (self.state.automap_buffer is None):
            return np.zeros((90, 160, 3), dtype=np.uint8)
        game_map = self.state.automap_buffer
        game_map = resize_frame(game_map.astype(np.uint8), 1/2)
        # game_map = cv2.cvtColor(game_map, cv2.COLOR_BGR2RGB)
        self.game_map_buffer.append(game_map)
        return game_map

    def get_padded_past_n_frames(self, n):
        past_n = np.zeros((n, 90, 160, 3), dtype=np.uint8)
        if self.frame_buffer:
            frames = list(self.frame_buffer)[-n:]
            past_n[-len(frames):] = frames
        # move first axis to last and merge with channel
        past_n = np.moveaxis(past_n, 0, -1)
        past_n = past_n.reshape((90, 160, 3*n))
        return  past_n
    def get_padded_past_n_gamemaps(self, n):
        past_n = np.zeros((n, 90, 160, 3), dtype=np.uint8)
        if self.game_map_buffer:
            frames = list(self.game_map_buffer)[-n:]
            past_n[-len(frames):] = frames
        # move first axis to last and merge with channel
        past_n = np.moveaxis(past_n, 0, -1)
        past_n = past_n.reshape((90, 160, 3*n))
        return  past_n
    
    def get_observation(self):
        past_32 = self.get_padded_last_32_actions()
        game_frames = self.get_padded_past_n_frames(4)
        game_maps = self.get_padded_past_n_gamemaps(4)
        obs = flatten_obs(game_frames, game_maps, past_32)
        return obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Clear buffers
        self.frame_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.count_by_class = {}    
        
        # Process initial observation
        processed_obs = resize_frame(obs['screen'], 1/2)
        self.frame_buffer.append(processed_obs)
        
        self.exploration_tracker = ExplorationTracker(40)
        self.past_action = None
        self.repeat_count = 0

        # Create observation
        past_32 = self.get_padded_last_32_actions()
        gamemap = self.get_gamemap()

        # frame = self.frame_buffer[-1]
        # obs = flatten_obs(processed_obs, gamemap, past_32)

        obs = self.get_observation()
        
        return obs, {}

    def step(self, action):
        reward = 0
        
        for _ in range(2):
            prev_state = self.state
            obs, _reward, terminated, truncated, info = self.env.step(action)

            processed_obs = resize_frame(obs['screen'], 1/2)
            self.frame_buffer.append(processed_obs)
            self.action_buffer.append(int(action))
            
            # Calculate reward
            if prev_state is not None and self.state is not None:
                fake_reward, rwd = calculate_reward(
                    self.game, 
                    prev_state.game_variables, 
                    self.state.game_variables, 
                    self.exploration_tracker
                )
                # for each item in rwd, add to count_by_class
                for _rwd in rwd:
                    if _rwd not in self.count_by_class:
                        self.count_by_class[_rwd] = 0
                    self.count_by_class[_rwd] += 1
                self.reward_buffer.append((fake_reward, rwd))
                reward += (fake_reward + _reward)
            
            if terminated or truncated:
                break
        reward = symlog(reward)
   
        # obs = flatten_obs(processed_obs, gamemap, past_32)
        obs = self.get_observation()
        # obs = flatten_obs(processed_obs, gamemap, past_32)

        if terminated or truncated:
            self.save_episode_data()
            
        return obs, reward, terminated, truncated, info

    def save_episode_data(self):
        self.episode_count += 1
        
        # # Only save every 2nd episode
        # if self.episode_count % 2 != 0 or not self.frame_buffer:
        #     self._clear_buffers()
        #     return
        
        try:
            # Process frames in chunks to reduce memory usage
            frames = list(self.frame_buffer)[::2]
            acts = list(self.action_buffer)[::2]
            rwd = list(self.reward_buffer)[::2]
            
            # Add height padding
            height_padding = np.zeros((24, frames[0].shape[1], 3), dtype=np.uint8)
            padded_frames = []
            
            # Process frames in chunks
            chunk_size = 100
            for i in range(0, len(frames), chunk_size):
                chunk = frames[i:i + chunk_size]
                padded_chunk = [np.concatenate([f, height_padding], axis=0) for f in chunk]
                padded_frames.extend(padded_chunk)
                del chunk
                gc.collect()
            
            base_filename = f"episode_{self.episode_count}_{int(time.time())}"
            video_filepath = os.path.join(self.video_folder, f"{base_filename}.mp4")
            json_filepath = os.path.join(self.video_folder, f"{base_filename}.json")

            # Create video with reduced memory usage
            clip = ImageSequenceClip(padded_frames, fps=18)
            clip.write_videofile(
                video_filepath,
                codec='libx264',
                verbose=False,
                logger=None,
                preset='ultrafast',
                ffmpeg_params=["-crf", "39", "-preset", "ultrafast"],
                fps=18,
                threads=8
            )
            
            # Save metadata
            episode_data = {
                "actions": acts,
                "rewards": rwd,
                "num_frames": len(frames),
                "timestamp": time.time(),
                "count_by_class": self.count_by_class
            }
            
            with open(json_filepath, 'w') as f:
                json.dump(episode_data, f)
                
        except Exception as e:
            print(f"Error saving episode: {e}")
        finally:
            self._clear_buffers()
            gc.collect()
    
    def _clear_buffers(self):
        """Helper method to clear all buffers"""
        self.frame_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.exploration_tracker.visited_positions = []
        self.count_by_class = {}

class DoomWrapperEps(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env.unwrapped)
        # if env.observation_space['screen'].shape[0] != 180:
        #     raise ValueError(
        #         'Wrong screen resolution. Doom does not provide '
        #         'a way to change this. You must edit scenarios/<env_name>.cfg.'
        #     )

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(OBS_SPACE,), dtype=np.uint8
        )
        
        self.video_folder = '/home/ubuntu/vizdoom/episodes'
        os.makedirs(self.video_folder, exist_ok=True)
        
        # Use deque with maxlen for better memory management
        from collections import deque
        self.frame_buffer = deque(maxlen=8)  # Limit frame buffer size
        self.game_map_buffer = deque(maxlen=8)  # Limit game map buffer size
        self.action_buffer = deque(maxlen=32)  # Limit action buffer size
        self.reward_buffer = deque(maxlen=32)  # Limit reward buffer size
        
        self.episode_count = 0
        self.exploration_tracker = ExplorationTracker(40)
        self.past_action = None
        self.repeat_count = 0
        self.count_by_class = {}

    def get_padded_last_32_actions(self):
        past_32 = np.zeros(32, dtype=np.uint8)
        if self.action_buffer:
            actions = list(self.action_buffer)[-32:]
            past_32[-len(actions):] = np.array(actions) + 1
        return past_32

    def get_gamemap(self):
        if (self.state is None) or (self.state.automap_buffer is None):
            return np.zeros((90, 160, 3), dtype=np.uint8)
        game_map = self.state.automap_buffer
        game_map = resize_frame(game_map.astype(np.uint8), 1/4)
        # game_map = cv2.cvtColor(game_map, cv2.COLOR_BGR2RGB)
        self.game_map_buffer.append(game_map)
        return game_map

    def get_padded_past_n_frames(self, n):
        past_n = np.zeros((n, 90, 160, 3), dtype=np.uint8)
        if self.frame_buffer:
            frames = list(self.frame_buffer)[-n:]
            past_n[-len(frames):] = frames
        # move first axis to last and merge with channel
        past_n = np.moveaxis(past_n, 0, -1)
        past_n = past_n.reshape((90, 160, 3*n))
        return  past_n
    def get_padded_past_n_gamemaps(self, n):
        past_n = np.zeros((n, 90, 160, 3), dtype=np.uint8)
        if self.game_map_buffer:
            frames = list(self.game_map_buffer)[-n:]
            past_n[-len(frames):] = frames
        # move first axis to last and merge with channel
        past_n = np.moveaxis(past_n, 0, -1)
        past_n = past_n.reshape((90, 160, 3*n))
        return  past_n
    
    def get_observation(self):
        past_32 = self.get_padded_last_32_actions()
        game_frames = self.get_padded_past_n_frames(4)
        game_maps = self.get_padded_past_n_gamemaps(4)
        obs = flatten_obs(game_frames, game_maps, past_32)
        return obs

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Clear buffers
        self.frame_buffer.clear()
        self.action_buffer.clear()
        # self.reward_buffer.clear()
        # self.count_by_class = {}    
        
        # Process initial observation
        processed_obs = resize_frame(obs['screen'], 1/4)
        self.frame_buffer.append(processed_obs)
        
        # self.exploration_tracker = ExplorationTracker(40)
        self.past_action = None
        self.repeat_count = 0


        obs = self.get_observation()
        
        return obs, {}

    def step(self, action):
        reward = 0

        env_vars = []
        
        for _ in range(2):
            prev_state = self.state
            obs, _reward, terminated, truncated, info = self.env.step(action)
            actual_obs= obs['screen']
            processed_obs = resize_frame(obs['screen'], 1/4)
            self.frame_buffer.append(processed_obs)
            self.action_buffer.append(int(action))
            if (self.state is not None) and (self.state.game_variables is not None):
                list_game_vars = list(self.state.game_variables)
                list_game_vars = [int(i) for i in list_game_vars]
                env_vars.append(list_game_vars)
            
            # Calculate reward
            # if prev_state is not None and self.state is not None:
            #     fake_reward, rwd = calculate_reward(
            #         self.game, 
            #         prev_state.game_variables, 
            #         self.state.game_variables, 
            #         self.exploration_tracker
            #     )
            #     # for each item in rwd, add to count_by_class
            #     for _rwd in rwd:
            #         if _rwd not in self.count_by_class:
            #             self.count_by_class[_rwd] = 0
            #         self.count_by_class[_rwd] += 1
            #     self.reward_buffer.append((fake_reward, rwd))
            #     reward += fake_reward
            reward = 0
            if terminated or truncated:
                break
        reward = symlog(reward)
   
        # obs = flatten_obs(processed_obs, gamemap, past_32)
        obs = self.get_observation()

        return obs, env_vars, terminated, truncated, info, actual_obs

    def _clear_buffers(self):
        """Helper method to clear all buffers"""
        self.frame_buffer.clear()
        self.action_buffer.clear()
        # self.reward_buffer.clear()
        # self.exploration_tracker.visited_positions = []
        self.count_by_class = {}


class ExplorationTracker:
    def __init__(self, distance_threshold=200):  # distance in Doom units
        self.distance_threshold = distance_threshold
        self.visited_positions = []
        self.total_distance = 0

    def add_position(self, x, y):
        current_pos = np.array([x, y])
        if not self.visited_positions:
            self.visited_positions.append(current_pos)
            return 0  # Reward for the first position

        # Calculate L1 (Manhattan) distances to all visited positions
        distances = [np.sum(np.abs(current_pos - pos)) for pos in self.visited_positions]
        min_distance = min(distances)

        if min_distance > self.distance_threshold:
            self.visited_positions.append(current_pos)
            self.total_distance += min_distance
            return min_distance

        return 0



def calculate_reward(game, prev_game_variables, current_game_variables, exploration_tracker):
    """
    Calculates the intrinsic reward based on various game events and exploration.
    """
    reasons = []

    # Define indices for game variables for readability
    KILL_COUNT = 0
    ITEM_COUNT = 1
    SECRET_COUNT = 2
    FRAG_COUNT = 3
    DEATH_COUNT = 4
    HIT_COUNT = 5
    HITS_TAKEN = 6
    DAMAGE_COUNT = 7
    DAMAGE_TAKEN = 8
    HEALTH = 9
    ARMOR = 10
    DEAD = 11
    SELECTED_WEAPON_AMMO = 12
    POSITION_X = 13
    POSITION_Y = 14
    POSITION_Z = 15

    reward = 0
    # 1. Player hit: -50 points (Reduced penalty)
    if current_game_variables[HITS_TAKEN] > prev_game_variables[HITS_TAKEN]:
        reward -= 100
        reasons.append("hit_taken")

    # 2. Player death: -2000 points (Reduced penalty)
    if current_game_variables[DEAD] > prev_game_variables[DEAD]:
        reward -= 5000
        reasons.append("player_death")

    # 3. Enemy hit: +150 points per hit (Reduced reward)
    damage_dealt = current_game_variables[DAMAGE_COUNT] - prev_game_variables[DAMAGE_COUNT]
    enemy_hits = damage_dealt  # Assuming each damage point corresponds to a hit
    if enemy_hits > 0:
        reward += 500 * enemy_hits
        reasons.append("enemy_hit")

    # 4. Enemy kill: +500 points per kill (Reduced reward)
    kills = current_game_variables[KILL_COUNT] - prev_game_variables[KILL_COUNT]
    if kills > 0:
        reward += 2500 * kills
        reasons.append("enemy_kill")

    # 5. Item/weapon pick up: +150 points per item (Increased reward)
    items_picked = current_game_variables[ITEM_COUNT] - prev_game_variables[ITEM_COUNT]
    if items_picked > 0:
        reward += 100 * items_picked
        reasons.append("item_pickup")

    # 6. Secret found: +300 points per secret (Reduced reward)
    secrets_found = current_game_variables[SECRET_COUNT] - prev_game_variables[SECRET_COUNT]
    if secrets_found > 0:
        reward += 500 * secrets_found
        reasons.append("secret_found")

    # 7. Exploration Reward: +10 points for visiting a new area
    curr_x = current_game_variables[POSITION_X]
    curr_y = current_game_variables[POSITION_Y]
    exploration_bonus = exploration_tracker.add_position(curr_x, curr_y)
    if exploration_bonus > 0:
        reward += 5 * (1 + (exploration_bonus/2.0))
        reasons.append("exploration_bonus")

    # 8. Health delta: +5 * delta for positive changes, -10 * delta for negative changes
    health_delta = current_game_variables[HEALTH] - prev_game_variables[HEALTH]
    reward += (10 * health_delta)
    if(abs(health_delta) > 0):
        reasons.append("health_change")

    # 9. Armor delta: +5 * delta for positive changes, -10 * delta for negative changes
    armor_delta = current_game_variables[ARMOR] - prev_game_variables[ARMOR]
    reward += (10 * armor_delta)
    reward += (10 * max(armor_delta, 0)) + (10 * min(armor_delta, 0))
    if(abs(armor_delta) > 0):
        reasons.append("armor_delta")

    # Additional Constraints to Prevent Reward Hacking
    # Example: Limit the maximum reward or penalty per step
    # max_reward = 1000
    # min_reward = -3000
    # reward = np.clip(reward, min_reward, max_reward)

    # reward scale
    # reward 
    # max 1000
    # min -1000
    reward = min(500, max(-500, reward))
    return reward, reasons


if __name__ == "__main__":

    env = gym.make("VizdoomOblige-v0")
    env.reset()

    # random policy
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(reward, done)
        if done:
            obs, info = env.reset()
            print("reset")
    env.close()