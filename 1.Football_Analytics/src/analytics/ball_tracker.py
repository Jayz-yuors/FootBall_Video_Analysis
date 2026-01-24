import cv2

# Ball color (Dark Maroon)
BALL_COLOR = (0, 0, 128)

class BallTracker:
    def __init__(self):
        self.last_ball = None  # Keep previous ball location if lost

    def update(self, ball_box):
        """Store latest ball box if detected"""
        if ball_box is not None:
            self.last_ball = ball_box

    def draw(self, frame):
        """Draw clean box around the ball if available"""
        if self.last_ball is None:
            return

        x1, y1, x2, y2 = map(int, self.last_ball)

        # Small bounding box around ball
        cv2.rectangle(frame, (x1, y1), (x2, y2), BALL_COLOR, 2)
