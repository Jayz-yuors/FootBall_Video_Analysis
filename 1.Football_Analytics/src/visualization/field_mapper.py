# src/visualization/field_mapper.py
from typing import Tuple
def map_to_field(
    x: float,
    y: float,
    frame_width: int,
    frame_height: int,
    field_width_m: float = 105.0,
    field_height_m: float = 68.0,
) -> Tuple[float, float]:
    """
    Map pixel coordinates (x, y) from video frame to normalized "field" coordinates in meters.

    Assumes:
    - The full frame roughly corresponds to the full pitch.
    This is a simplification; real calibration would need homography.

    returns: (field_x_meters, field_y_meters)
    """
    if frame_width <= 0 or frame_height <= 0:
        return 0.0, 0.0

    norm_x = x / frame_width   # 0..1
    norm_y = y / frame_height  # 0..1

    field_x = norm_x * field_width_m
    field_y = norm_y * field_height_m

    return field_x, field_y
