import numpy as np
import cv2
from typing import List,Tuple ,Dict
def generate_position_heatmap(
        history : Dict[int,List[Tuple[int,float,float]]],
        width : int,
        height : int,
        kernel_size : int = 25,
) -> np.ndarrray:
    """
    Generate a simple heatmap (2D array) of all player positions combined.

    history: {track_id: [(frame_idx, x, y), ...]}
    width, height: frame dimensions
    kernel_size: size of blur kernel to smooth heatmap
    returns: heatmap as a HxW uint8 image (0-255)
    """
    heatmap = np.zeros((height,width),dtype = np.float32)
    for _,points in history.items():
        for _,x,y in points:
            ix = int(np.clip(x, 0, width - 1))
            iy = int(np.clip(y, 0, height - 1))
            heatmap[iy,ix] += 1.0
    # Normalize heatmap to 0-255
    if heatmap.max() > 0 :
        heatmap /= heatmap.max()
        heatmap *= 255.0
    heatmap = heatmap.astype(np.uint8) #Convert heatmap to 8-bit unsigned integers, values 0–255.
    if kernel_size > 1 :
        heatmap = cv2.GaussianBlur(heatmap,(kernel_size,kernel_size),0) #Smoothens the heatmap
    return heatmap
def apply_colormap_to_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """
    Converts a single-channel heatmap into a color heatmap.
    """
    color_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return color_map
