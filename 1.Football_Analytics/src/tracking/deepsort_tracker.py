# src/tracking/deepsort_tracker.py

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    """
    Wrapper around deep_sort_realtime.DeepSort to work with YOLOv8 detections.

    Expected input to update():
        detections: List of (x1, y1, x2, y2, score)
        frame:      BGR image (numpy array)
    """

    def __init__(self):
        # You can tune these hyperparameters later
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=0.7,
            max_iou_distance=0.7,
            # keep default embedder="mobilenet" etc.
        )

    def update(self, detections, frame):
        """
        Convert YOLO [x1, y1, x2, y2, score] → DeepSORT format:
        ([left, top, w, h], confidence, class_name)
        and call update_tracks()
        """

        # Build detection list for DeepSort:
        # each element: ([l, t, w, h], conf, class_str)
        deep_sort_dets = []

        for det in detections:
            x1, y1, x2, y2, score = det
            w = x2 - x1
            h = y2 - y1

            # DeepSORT expects this tuple:
            # ([left, top, width, height], confidence, detection_class)
            deep_sort_dets.append(
                ([float(x1), float(y1), float(w), float(h)], float(score), "person")
            )

        # Call DeepSORT
        tracks = self.tracker.update_tracks(deep_sort_dets, frame=frame)

        # Convert back to: (track_id, (x1, y1, x2, y2)) for our pipeline
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()  # left, top, right, bottom

            results.append(
                (int(track_id), (float(l), float(t), float(r), float(b)))
            )

        return results
