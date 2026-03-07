import numpy as np
from collections import deque

class TrackingLayer:
    def __init__(self, config):
        self.history_len = config['tracking']['max_trajectory_history']
        self.alpha = 0.3 # Smoothing factor for 3D coordinates
        
        # Dict mapping track_id -> deque of bounding boxes/centers over time
        self.trajectories = {}
        
    def update(self, tracked_objects):
        """
        Takes the tracked current frame objects and updates their state queue.
        Calculates centroid.
        """
        current_ids = set()
        
        for obj in tracked_objects:
            t_id = obj['id']
            bbox = obj['bbox']
            cls_id = obj['class']
            
            # Simple centroid x, y and pseudo-size (box area proxy)
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Structure defining a historical point
            bev_raw = obj.get('bev', [0, 0])
            
            # Apply EMA Smoothing to BEV to prevent HUD jitter
            if t_id in self.trajectories and len(self.trajectories[t_id]) > 0:
                prev_bev = self.trajectories[t_id][-1]['bev']
                bev_smoothed = [
                    self.alpha * bev_raw[0] + (1 - self.alpha) * prev_bev[0],
                    self.alpha * bev_raw[1] + (1 - self.alpha) * prev_bev[1]
                ]
            else:
                bev_smoothed = bev_raw

            point = {
                'cx': cx,
                'cy': cy,
                'bbox': bbox,
                'size': size,
                'class': cls_id,
                'bev': bev_smoothed
            }
            
            if t_id not in self.trajectories:
                self.trajectories[t_id] = deque(maxlen=self.history_len)
                
            self.trajectories[t_id].append(point)
            current_ids.add(t_id)
            
        # Clean up lost tracks to avoid memory leaks
        lost_ids = set(self.trajectories.keys()) - current_ids
        for lid in lost_ids:
            # For robustness, we could keep them for a few frames before destroying
            # but for this research demo, we discard immediately to avoid ghost trajectories
            del self.trajectories[lid]
            
        return self.trajectories
