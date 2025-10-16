# 2D to 3D F1 Track Generator

This script converts a 2D PNG image of an F1 race track (black track on white background) into a 3D environment in PyBullet. It detects the track boundaries, smooths them, and generates 3D walls and a road surface for simulation or visualization.

## Features
- Loads a PNG image of a track
- Extracts and smooths track boundaries
- Creates 3D walls along both sides of the track
- Generates a 3D road mesh using trimesh
- Visualizes the track in PyBullet

## Requirements
- Python 3.11.11 (3,8+ should also work)
- stuff in requirements.txt

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage
1. Place your PNG track image (black-ish track, white-ish background) in the working directory.
2. Run the script, to generate the 3D track and visualize it in PyBullet:
   ```sh
   python 2dTo3d_RoadTrack.py track.png
   ```
    Addionally, to specify output file for the obj model:
   ```
   python 2dTo3d_RoadTrack.py track.png output.obj
   ```

   Replace `track.png` with your image filename.
3. The PyBullet window will open with the generated 3D track.

## Notes
- Adjust smoothing and scaling parameters in the script for best results.
- The script expects the track to be the largest black region in the image.
