import os
import cv2
import numpy as np
import csv
import torch

class VideoTensorStorage:
    def __init__(self, max_video_frames, grid_size, frame_height, frame_width, num_envs):
        self.max_video_frames = max_video_frames
        self.grid_size = grid_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_envs = num_envs
        self.video_file_count = 0
        self.frame_count = 0
        self.video_writer = None
        self.episode_tracker = []
        self.episode_counters = torch.zeros((num_envs,), dtype=torch.int32)  # Track episode numbers for each environment
        self.video_paths = []  # To store the paths of all video files for reading later
        self.csv_paths = []    # To store the paths of all episode CSV files

        # Create directory for storing videos and CSVs
        os.makedirs('trajectory_videos', exist_ok=True)

        # Open the first video writer
        self.open_video_writer()

    def open_video_writer(self):
        """Open a new video file for writing frames."""
        self.video_file_count += 1
        video_path = os.path.join("trajectory_videos", f"frames_{self.video_file_count}.mp4")
        self.video_paths.append(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            video_path, fourcc, 20.0, (self.frame_width * self.grid_size, self.frame_height * self.grid_size)
        )

    def close_video_writer(self):
        """Close the current video writer."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def save_episode_csv(self):
        """Save the episode tracking information to a CSV file."""
        csv_path = os.path.join(f"trajectory_videos", f"episodes_{self.video_file_count}.csv")
        self.csv_paths.append(csv_path)
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.episode_tracker)

    def update_and_save_frame(self, observations, done_flags):
        """Update the video with a new frame and track episode info."""
        # Create the tiled video frame grid
        frames = observations.cpu().numpy()  # Assuming observations are the frames in shape (NUM_ENVS, H, W, C)
        grid_frame = np.zeros((self.frame_height * self.grid_size, self.frame_width * self.grid_size, 3), dtype=np.uint8)

        for i in range(self.num_envs):
            row = i // self.grid_size
            col = i % self.grid_size
            grid_frame[
                row * self.frame_height:(row + 1) * self.frame_height,
                col * self.frame_width:(col + 1) * self.frame_width
            ] = frames[i]

        # Write the tiled frame to the video
        self.video_writer.write(cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR))

        # Update the episode tracker for each frame
        self.episode_tracker.append(self.episode_counters.clone().tolist())

        # Handle done flags and increment episode counters
        for i, done in enumerate(done_flags):
            if done:
                self.episode_counters[i] += 1

        # Manage video file splitting
        self.frame_count += 1
        if self.frame_count >= self.max_video_frames:
            self.close_video_writer()
            self.save_episode_csv()  # Save the CSV file for the current video segment
            self.open_video_writer()
            self.frame_count = 0
            self.episode_tracker = []  # Reset tracker for the next chunk of video

    def get_video_slice(self, env_i: int, episode: int):
        """Extract a slice of video for the specified environment and episode."""
        # Get the row/column of the environment in the tiled video grid
        row = env_i // self.grid_size
        col = env_i % self.grid_size

        # Determine the x/y coordinates of the tile
        x_start = col * self.frame_width
        x_end = x_start + self.frame_width
        y_start = row * self.frame_height
        y_end = y_start + self.frame_height

        # Step 1: Collect relevant frames for the given episode
        episode_frames = []
        for csv_path, video_path in zip(self.csv_paths, self.video_paths):
            # Load the CSV file to find relevant frames
            with open(csv_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for frame_idx, row in enumerate(reader):
                    # Check if the episode for env_i matches the requested episode
                    if int(row[env_i]) == episode:
                        episode_frames.append((frame_idx, video_path))

        # Step 2: Pre-allocate a tensor to hold the frames
        num_frames = len(episode_frames)
        video_tensor = torch.zeros((num_frames, 3, self.frame_height, self.frame_width), dtype=torch.uint8)

        # Step 3: Read frames from the video, extract the tile, and fill the tensor
        for i, (frame_idx, video_path) in enumerate(episode_frames):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)  # Move to the correct frame
            ret, frame = cap.read()

            if ret:
                # Extract the environment's tile from the frame
                tile = frame[y_start:y_end, x_start:x_end]
                video_tensor[i] = torch.from_numpy(tile).permute(2, 0, 1)  # Convert to torch tensor (C, H, W)
            cap.release()

        return video_tensor

    def close(self):
        """Finalize the storage by closing the video writer and saving the last episode CSV."""
        self.close_video_writer()
        self.save_episode_csv()  # Save the CSV for the last video segment
