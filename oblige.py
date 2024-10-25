import gymnasium
from vizdoom import gymnasium_wrapper


from gymnasium.envs.registration import register
import os

# from oblige import make_oblige


scenario_file = os.path.join(os.path.dirname(__file__), "scenarios", "oblige_custom.cfg")
register(
    id="VizdoomOblige-v0",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_file":scenario_file},
)




def get_state(game):

    # compose
    KILL_COUNT = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
    ITEM_COUNT = game.get_game_variable(vzd.GameVariable.ITEMCOUNT)
    SECRET_COUNT = game.get_game_variable(vzd.GameVariable.SECRETCOUNT)
    FRAG_COUNT = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
    DEATH_COUNT = game.get_game_variable(vzd.GameVariable.DEATHCOUNT)
    HIT_COUNT = game.get_game_variable(vzd.GameVariable.HITCOUNT)
    HITS_TAKEN = game.get_game_variable(vzd.GameVariable.HITS_TAKEN)
    DAMAGE_COUNT = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
    DAMAGE_TAKEN = game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
    HEALTH = game.get_game_variable(vzd.GameVariable.HEALTH)
    ARMOR = game.get_game_variable(vzd.GameVariable.ARMOR)
    DEAD = game.get_game_variable(vzd.GameVariable.DEAD)
    SELECTED_WEAPON_AMMO = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
    POSITION_X = game.get_game_variable(vzd.GameVariable.POSITION_X)
    POSITION_Y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
    POSITION_Z = game.get_game_variable(vzd.GameVariable.POSITION_Z)
    return (KILL_COUNT, ITEM_COUNT, SECRET_COUNT, FRAG_COUNT, DEATH_COUNT, HIT_COUNT, HITS_TAKEN, DAMAGE_COUNT, DAMAGE_TAKEN, HEALTH, ARMOR, DEAD, SELECTED_WEAPON_AMMO, POSITION_X, POSITION_Y, POSITION_Z)


def get_reward(prev_state, next_state):
    (PREV_KILL_COUNT, PREV_ITEM_COUNT, PREV_SECRET_COUNT, PREV_FRAG_COUNT, PREV_DEATH_COUNT, PREV_HIT_COUNT, PREV_HITS_TAKEN, PREV_DAMAGE_COUNT, PREV_DAMAGE_TAKEN, PREV_HEALTH, PREV_ARMOR, PREV_DEAD, PREV_SELECTED_WEAPON_AMMO, PREV_POSITION_X, PREV_POSITION_Y, PREV_POSITION_Z) = prev_state
    (NEXT_KILL_COUNT, NEXT_ITEM_COUNT, NEXT_SECRET_COUNT, NEXT_FRAG_COUNT, NEXT_DEATH_COUNT, NEXT_HIT_COUNT, NEXT_HITS_TAKEN, NEXT_DAMAGE_COUNT, NEXT_DAMAGE_TAKEN, NEXT_HEALTH, NEXT_ARMOR, NEXT_DEAD, NEXT_SELECTED_WEAPON_AMMO, NEXT_POSITION_X, NEXT_POSITION_Y, NEXT_POSITION_Z) = next_state

    reward = 0
    #1. Player hit: -100 points.  2. Player death: -5,000 points.  3. Enemy hit: 300 points.  4. Enemy kill: 1,000 points.  5. Item/weapon pick up: 100 points.  6. Secret found: 500 points.  7. New area: 20 * (1 + 0.5 * L1 distance) points.  8. Health delta: 10 * delta points.  9. Armor delta: 10 * delta points.  10. Ammo delta: 10 * max(0, delta) + min(0, delta) points.
    reward += (NEXT_HIT_COUNT - PREV_HIT_COUNT) * 300
    reward += (NEXT_KILL_COUNT - PREV_KILL_COUNT) * 1000
    reward += (NEXT_ITEM_COUNT - PREV_ITEM_COUNT) * 100
    reward += (NEXT_SECRET_COUNT - PREV_SECRET_COUNT) * 500
    # reward += (NEXT_POSITION_X - PREV_POSITION_X) * 20
    # reward += (NEXT_POSITION_Y - PREV_POSITION_Y) * 20
    # reward += (NEXT_POSITION_Z - PREV_POSITION_Z) * 20
    reward += (NEXT_HEALTH - PREV_HEALTH) * 10
    reward += (NEXT_ARMOR - PREV_ARMOR) * 10
    reward += (NEXT_SELECTED_WEAPON_AMMO - PREV_SELECTED_WEAPON_AMMO) * 10
    return reward

if __name__ == "__main__":

    import vizdoom as vzd

    # env = gymnasium.make("VizdoomDefendCenter-v0")
    env = gymnasium.make("VizdoomOblige-v0")
    # env = make_oblige("VizdoomOblige-v0")

    game = env.env.env.game
    state = game.get_state()

    print(game)
    print(state)

    # print(get_reward(game, state))

    # exit()

    observation, info = env.reset()

    # let's do a cv2 window to show the screen
    import cv2
    cv2.namedWindow("screen", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("screen", 640, 480)

    for _ in range(1000000):
        # NOTE: only one action is valid at a time
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
        prev_state = get_state(game)

        observation, _, terminated, truncated, info = env.step(action)
        next_state = get_state(game)

        reward = get_reward(prev_state, next_state)
        # print(env.action_space)
        # if reward != 0:
        print(reward)
        # print(observation["screen"].shape, reward)
        # print(observation["screen"].dtype)
        # print(observation["gamevariables"].shape, reward)

        # show the screen
        cv2.imshow("screen", observation["screen"])
        cv2.waitKey(1)

        if terminated or truncated:
            observation, info = env.reset()


    env.close()