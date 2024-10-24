import os
import cv2
import numpy as np
import csv
import torch

class VideoTensorStorage:
    def __init__(self, subdirectory: str, max_video_frames, grid_size, frame_height, frame_width, num_envs):
        self.max_video_frames = max_video_frames
        self.grid_size = grid_size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_envs = num_envs
        self.video_file_count = 0
        self.frame_count = 0
        self.video_writer = None
        self.episode_tracker = []
        self.unsaved_episode_tracker = []  # New: Track unsaved episode data
        self.episode_counters = torch.zeros((num_envs,), dtype=torch.int32)
        self.video_paths = []
        self.csv_paths = []

        self.folder = os.path.join("trajectory_videos", subdirectory)
        os.makedirs(self.folder, exist_ok=True)

        self.open_video_writer()

    def open_video_writer(self):
        """Open a new video file for writing frames."""
        self.video_file_count += 1
        video_path = os.path.join(self.folder, f"frames_{self.video_file_count}.mp4")
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
        csv_path = os.path.join(self.folder, f"episodes_{self.video_file_count}.csv")
        self.csv_paths.append(csv_path)
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.episode_tracker)

        # Clear the in-memory tracker since it's now saved
        self.unsaved_episode_tracker = []

    def update_and_save_frame(self, observations, done_flags):
        # Same logic for updating video frames
        frames = observations.cpu().numpy()
        grid_frame = np.zeros((self.frame_height * self.grid_size, self.frame_width * self.grid_size, 3), dtype=np.uint8)

        for i in range(self.num_envs):
            row = i // self.grid_size
            col = i % self.grid_size
            grid_frame[row * self.frame_height:(row + 1) * self.frame_height, col * self.frame_width:(col + 1) * self.frame_width] = frames[i]

        self.video_writer.write(cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR))

        self.episode_tracker.append(self.episode_counters.clone().tolist())
        self.unsaved_episode_tracker.append(self.episode_counters.clone().tolist())  # Also track in-memory unsaved data

        for i, done in enumerate(done_flags):
            if done:
                self.episode_counters[i] += 1

        self.frame_count += 1
        if self.frame_count >= self.max_video_frames:
            self.close_video_writer()
            self.save_episode_csv()
            self.open_video_writer()
            self.frame_count = 0
            self.episode_tracker = []

    def _clip_current_chunk(self):
        """
        Finalize the current video capture by closing the VideoWriter and saving the episode CSV.
        This will allow you to manually clip the current video and start a new one.
        """

        # Close the current video writer and finalize the CSV for the current segment
        self.close_video_writer()
        self.save_episode_csv()
        
        # Reset frame count and episode tracker for the next segment
        self.frame_count = 0
        self.episode_tracker = []
        
        # Open a new video writer for the next video segment
        self.open_video_writer()

    def get_video_slice(self, env_i: int, episode: int):
        """Will clip the video before filling the full videos if it's partially through a chunk.
        """

        if len(self.unsaved_episode_tracker) > 0:
            self._clip_current_chunk()

        row = env_i // self.grid_size
        col = env_i % self.grid_size
        x_start = col * self.frame_width
        x_end = x_start + self.frame_width
        y_start = row * self.frame_height
        y_end = y_start + self.frame_height

        episode_frames = []

        # Check the CSV paths for frames associated with the episode
        for csv_path, video_path in zip(self.csv_paths, self.video_paths):
            with open(csv_path, newline='') as csvfile:
                reader = csv.reader(csvfile)
                for frame_idx, row in enumerate(reader):
                    if int(row[env_i]) == episode:
                        episode_frames.append((frame_idx, video_path))
        
        # Check unsaved data first
        for frame_idx, tracker_row in enumerate(self.unsaved_episode_tracker):
            if int(tracker_row[env_i]) == episode:
                episode_frames.append((self.frame_count - len(self.unsaved_episode_tracker) + frame_idx, self.video_paths[-1]))

        print(f"Found {len(episode_frames)} frames for environment {env_i}, episode {episode}")
        if len(episode_frames) == 0:
            raise ValueError(f"No frames found for environment {env_i}, episode {episode}")

        video_tensor = torch.zeros((len(episode_frames), 3, self.frame_height, self.frame_width), dtype=torch.uint8)

        # Create a dictionary to cache cv2.VideoCapture instances for reuse
        capture_cache = {}

        for i, (frame_idx, video_path) in enumerate(episode_frames):
            # Reuse VideoCapture if already opened for the same video_path
            if video_path not in capture_cache:
                capture_cache[video_path] = cv2.VideoCapture(video_path)

            cap = capture_cache[video_path]
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                tile = frame[y_start:y_end, x_start:x_end]
                video_tensor[i] = torch.from_numpy(tile).permute(2, 0, 1)

        # Release all VideoCapture objects
        for cap in capture_cache.values():
            cap.release()

        return video_tensor

    def close(self):
        """Finalize the storage by closing the video writer and saving the last episode CSV."""
        self.close_video_writer()
        self.save_episode_csv()  # Save the CSV for the last video segment
