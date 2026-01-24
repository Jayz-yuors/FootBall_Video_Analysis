import math
from typing import Dict,List,Tuple
def detect_high_speed_events(
        speed_history : Dict[int,List[Tuple[int,float]]],
        sprint_threshold : float = 7.0,
) -> Dict[int,List[Tuple[int,float]]]:
    """
    Detect "sprint" events based on speed threshold.
    speed_history: {track_id: [(frame_idx, speed_mps), ...]}
    returns: {track_id: [frame_idx_of_sprint, ...]}
    """
    sprints = {}
    for track_id,speeds in speed_history.items():
        sprint_frames = [
            frame_idx for frame_idx,speed in speeds if speed >= sprint_threshold
        ]
        sprints[track_id] = sprint_frames
    return sprints
def detect_direction_changes(
    history: Dict[int, List[Tuple[int, float, float]]],
    angle_threshold_deg: float = 45.0,
) -> Dict[int, List[int]]:
    """
    Detect significant direction changes (could be used as rough 'dribble' or 'cut' events).
    history: {track_id: [(frame_idx, x, y), ...]}
    returns: {track_id: [frame_idx_where_change_occurs, ...]}
    """
    direction_events = {}

    for track_id, points in history.items():
        events = []
        if len(points) < 3:
            direction_events[track_id] = events
            continue

        for i in range(2, len(points)):
            _, x0, y0 = points[i - 2]
            _, x1, y1 = points[i - 1]
            frame_idx, x2, y2 = points[i]

            v1x, v1y = x1 - x0, y1 - y0
            v2x, v2y = x2 - x1, y2 - y1

            mag1 = math.hypot(v1x, v1y)
            mag2 = math.hypot(v2x, v2y)
            if mag1 == 0 or mag2 == 0:
                continue

            dot = v1x * v2x + v1y * v2y
            cos_angle = max(min(dot / (mag1 * mag2), 1.0), -1.0)
            angle = math.degrees(math.acos(cos_angle))

            if angle >= angle_threshold_deg:
                events.append(frame_idx)

        direction_events[track_id] = events

    return direction_events