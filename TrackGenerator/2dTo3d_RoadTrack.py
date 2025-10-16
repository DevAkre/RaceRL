# import required libraries
import cv2
import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import trimesh
from shapely.geometry import Polygon 
import sys

def load_track_mask(image_path):
    """Load PNG and return a binary mask where track is 1, background is 0."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    # Threshold: track is black (0), background is white (255)
    _, mask = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
    return mask




def mask_to_contours(mask, n=1, smooth_factor=0.003):
    """Find all significant contours (boundaries) of the track mask, with smoothing."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(f"Found {len(contours)} raw contours.")
    if not contours:
        raise ValueError("No contours found in mask.")
    # Sort contours by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    min_area = 100  # adjust as needed
    filtered = []
    for c in contours:
        if cv2.contourArea(c) > min_area:
            epsilon = smooth_factor * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            filtered.append(approx[:, 0, :][::n])  # sample every nth point if needed
    return filtered

def create_pybullet_walls(contours, wall_height=1.0, wall_thickness=0.1, scale=0.05):
    """Create walls along all track boundaries in PyBullet."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    for contour in contours:
        n = len(contour)
        for i in range(n):
            pt1 = contour[i]
            pt2 = contour[(i+1)%n]
            # Convert image to world coordinates
            x1, y1 = pt1[0] * scale, pt1[1] * scale
            x2, y2 = pt2[0] * scale, pt2[1] * scale
            # Wall segment midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            # Wall length and angle
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            angle = np.arctan2(dy, dx)
            # Create wall box
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_height/2])
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[length/2, wall_thickness/2, wall_height/2], rgbaColor=[1,0,0,1])
            orn = p.getQuaternionFromEuler([0, 0, angle])
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[mx, my, wall_height/2], baseOrientation=orn)
    return

def create_pybullet_road(contours, road_height=0.0, scale=0.05):
    """Create a flat road surface inside the track boundaries in PyBullet."""
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    # Create a road between the two contours
    if len(contours) < 2:
        raise ValueError("Need at least two contours to create a road.")
    outer, inner = contours[0], contours[1]
    # Create a mesh for the road surface
    # Convert contours to Nx2 float arrays
    outer = np.array(outer, dtype=np.float32)
    inner = np.array(inner, dtype=np.float32)[::-1]  # reverse inner for correct winding

    # Create a shapely polygon with a hole
    road_poly = Polygon(outer, [inner])

    # Use trimesh to extrude the polygon
    road_mesh = trimesh.creation.extrude_polygon(road_poly, height=0.01)

    # Move mesh so bottom is at road_height
    road_mesh.apply_translation([0, 0, road_height])

    # Export mesh to a temporary .obj file
    tmp_obj = ".tmp/track.obj"
    road_mesh.export(tmp_obj)

    # Load mesh into PyBullet as a visual and collision shape
    col_id = p.createCollisionShape(p.GEOM_MESH, fileName=tmp_obj, meshScale=[scale, scale, 1])
    vis_id = p.createVisualShape(p.GEOM_MESH, fileName=tmp_obj, meshScale=[scale, scale, 1], rgbaColor=[0.5,0.5,0.5,1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[0,0,road_height])
    
    return  

def visualize_contours(mask, contours):
    """Visualize contours on the mask using OpenCV."""
    vis_img = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) * 255
    cv2.drawContours(vis_img, [c.astype(np.int32) for c in contours], -1, (0,0,255),1)
    cv2.imshow("Contours", vis_img)
    cv2.waitKey(1)


def center_tracks(contours):
    """Center all track contours around the origin."""
    all_points = np.concatenate(contours, axis=0)
    centroid = np.mean(all_points, axis=0)
    return [c - centroid for c in contours]

def save_tracks(output_path):
    """Save model to a .obj file."""
    # Copy the temporary .obj to a permanent file
    src = ".tmp/track.obj"
    dst = output_path
    if os.path.exists(src):
        os.replace(src, dst)
        print(f"Model saved to {dst}")
    else:
        print("No model to save.")
    

def make3dObj(img_path, output_path=None):
    mask = load_track_mask(img_path)
    contours = mask_to_contours(mask)
    print(len(contours), "contours found.")
    centered = center_tracks(contours)
    # Show contour visualization
    visualize_contours(mask, contours)
    # create_pybullet_walls(centered, wall_height=0.5, wall_thickness=0.005, scale=0.05)
    create_pybullet_road(centered)
    # Ask to save contours
    if output_path:
        save_tracks(output_path)
    print("Press Ctrl+C in the PyBullet window to exit.")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        make3dObj(sys.argv[1])
    elif len(sys.argv) == 3:
        make3dObj(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python 2dTo3d_RoadTrack.py <image_path>")
        sys.exit(1)
