from interactor import DoomInteractor



if __name__ == "__main__":
    MAX_STEPS = 100
    NUM_ENVS = 16
    
    # if true one of the environments will be displayed in a cv2 window
    WATCH = False
    
    interactor = DoomInteractor(NUM_ENVS, watch=WATCH)

    # Reset all environments
    observations = interactor.env.reset()
    # print("Initial Observations:", observations.shape)

    # Example of stepping through the environments
    for _ in range(100):  # Step for 100 frames or episodes
        observations, rewards, dones = interactor.step()
        print(observations.shape, rewards.shape)

    # Close all environments
    interactor.env.close()
