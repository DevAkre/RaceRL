import cv2
import numpy as np
from typing import List

def prepare_frame(frame, velocity: float = 0.0, time: float = 0.0, motor: int = 0, motor_bins: int = 3, steer: int = 2, steer_bins:int = 5) -> np.ndarray:
    arr = np.asarray(frame)
    # Clip and convert to uint8 (handles int64, float, etc.)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
   
    # Convert RGB to BGR for OpenCV
    bgr_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    frame_height, frame_width, _ = bgr_frame.shape
    # Overlay velocity text bottom right corner
    cv2.putText(bgr_frame, f"{velocity:.2f} m/s", (frame_width - 100, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # Overlay time text bottom left corner
    cv2.putText(bgr_frame, f"Time: {time:.1f}s", (10, frame_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    # Show motor as a bar at the bottom center
    motor_bar_length = frame_width // 4
    motor_bar_height = frame_height // 10
    motor_x = int((frame_width - motor_bar_length) / 2)
    motor_y = frame_height - motor_bar_height - 10
    cv2.rectangle(bgr_frame, (motor_x, motor_y), (motor_x + motor_bar_length, motor_y + motor_bar_height), (200, 200, 200), 1)
    motor_fill_length = int((motor / (motor_bins-1)) * motor_bar_length) 
    cv2.rectangle(bgr_frame, (motor_x, motor_y), (motor_x + motor_fill_length, motor_y + motor_bar_height), (0, 255, 0), -1)
    # Label Brake/Neutral/Forward
    cv2.putText(bgr_frame, "B", (motor_x, motor_y + motor_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, "N", (motor_x + motor_bar_length // 2 - 5, motor_y + motor_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, "F", (motor_x + motor_bar_length - 7, motor_y + motor_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    # Show steering as a bar above the motor bar
    steer_bar_length = frame_width // 4
    steer_bar_height = frame_height // 10
    steer_x = int((frame_width - steer_bar_length) / 2)
    steer_y = motor_y - steer_bar_height
    cv2.rectangle(bgr_frame, (steer_x, steer_y), (steer_x + steer_bar_length, steer_y + steer_bar_height), (200, 200, 200), 1)
    steer_fill_length = int((steer / (steer_bins-1)) * steer_bar_length) 
    cv2.rectangle(bgr_frame, (steer_x, steer_y), (steer_x + steer_fill_length, steer_y + steer_bar_height), (255, 123, 123), -1)
    # Label Left/Right
    cv2.putText(bgr_frame, "L", (steer_x, steer_y + steer_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bgr_frame, "R", (steer_x + steer_bar_length - 7, steer_y + steer_bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return bgr_frame

def display_frame(frame, velocity: float = 0.0, time: float = 0.0):
    """Display a frame using OpenCV."""
    # Overlay steering and motor values top left corner
    cv2.imshow('Racecar', frame)
    cv2.waitKey(1)

def save_frames_to_video(frames: List[np.ndarray], output_path: str, fps: int = 30):
    """Save a list of frames to an MP4 video using OpenCV VideoWriter.
    
    Args:
        frames: List of RGB frames as numpy arrays
        output_path: Path to save the output video file
        fps: Frames per second for the output video
    """
    if not frames:
        print("No frames to save!")
        return
    
    print(f"Creating video with OpenCV at {output_path}...")
    
    # Get frame dimensions from first frame
    first_frame =  frames[0]
    height, width = first_frame.shape[:2]
    
    # Define the codec and create VideoWriter object
    # Use mp4v codec for MP4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # pyright: ignore[reportAttributeAccessIssue]
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_path}")
        return
    
    # Write each frame to the video
    print(f"Writing {len(frames)} frames to video...")
    for i, frame in enumerate(frames):
        arr = np.asarray(frame)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    
    # Release the VideoWriter
    out.release()
    print(f"Video saved successfully to {output_path}")
