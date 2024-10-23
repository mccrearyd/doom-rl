from doom_vec import play_doom





if __name__ == "__main__":
    NUM_ENVS = 8
    MAX_FRAMES = 100

    # obs shape (240, 320, 3)
    # uint8
    play_doom(num_envs=NUM_ENVS, max_frames=MAX_FRAMES)