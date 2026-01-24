import math 
from typing import Dict,List,Tuple
def compute_distance_travelled(
        history : Dict[int,List[Tuple[int,float,float]]]) -> Dict[int,float]:
    """
    Compute total distance travelled per track ID in pixel units.

    history: {track_id: [(frame_idx, x, y), ...]}
    returns: {track_id: distance_in_pixels}
    """
    distances = {}
    for track_id,points in history.items():
        if len(points) < 2 :
            distances[track_id] = 0.0
            continue
        dist = 0.0
    for i in range(1,len(points)):
        _,x1,y1 = points[i-1]
        _,x2,y2 = points[i]
        dist += math.hypot(x2 - x1, y2 - y1)
    distances[track_id] = dist
    return distances
def compute_speed_per_frame(
        history: Dict[int, List[Tuple[int, float, float]]],
    fps: float,
    pixel_to_meter_ratio: float = 1.0,
) -> Dict[int, List[Tuple[int, float]]]:
    """
    Compute approximate speed per frame (m/s) for each player.
    history: {track_id: [(frame_idx, x, y), ...]}
    fps: frames per second of the video
    pixel_to_meter_ratio: conversion factor from pixel-distance to meters
    returns: {track_id: [(frame_idx, speed_mps), ...]}
    """
    speeds = {}
    if fps <= 0 :
        fps = 25.0
    dt = 1.0 /fps
    for track_id,points in history.itmes():
        track_speeds = []
        if len(points) < 2 :
            speeds[track_id] = track_speeds
            continue
        for i in range(1,len(points)):
            frame_idx, x2, y2 = points[i]
            _, x1, y1 = points[i - 1] # _ is throwaway variable 
            distance_pixels = math.hypot(x2 - x1, y2 - y1)
            distance_meters = distance_pixels * pixel_to_meter_ratio
            speed_mps = distance_meters / dt
            track_speeds.append((frame_idx, speed_mps))
        speeds[track_id] = track_speeds
    return speeds

