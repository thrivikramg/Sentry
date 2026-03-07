import numpy as np
import math

class TrajectoryAnalyzer:
    def __init__(self, config):
        pass  # No direct configs needed yet, but we accept it for future expansion.

    def _calc_looming(self, p0, p2):
        """Calculates 3D BEV Time-to-Collision (if available), else 2D proxy."""
        if 'bev' in p0 and 'bev' in p2 and p0['bev'][1] != 0:
            z0 = p0['bev'][1]
            z2 = p2['bev'][1]
            # Velocity towards ego (Z is distance, decreasing Z means approaching)
            vz = z0 - z2
            
            # Growth proxy (vz)
            growth = vz
            ttc = z2 / max(0.01, vz) if vz > 0.05 else 999.0
            return growth, ttc
            
        # Fallback to 2D proxy
        a0 = (p0['bbox'][2] - p0['bbox'][0]) * (p0['bbox'][3] - p0['bbox'][1])
        a2 = (p2['bbox'][2] - p2['bbox'][0]) * (p2['bbox'][3] - p2['bbox'][1])
        s1, s2 = math.sqrt(max(1.0, a0)), math.sqrt(max(1.0, a2))
        growth = (s2 - s1)
        ttc = s2 / growth if growth > 0.1 else 999.0
        return growth, ttc

    def _calc_lateral(self, p1, p2):
        """Analyzes horizontal movement and lane convergence."""
        vx = p2['cx'] - p1['cx']
        is_moving_to_center = (p2['cx'] < 320 and vx > 0) or (p2['cx'] > 320 and vx < 0)
        return vx, is_moving_to_center

    def compute_dynamics(self, trajectories, class_names=None):
        """
        Calculates high-level behavioral features for every tracked object.
        """
        dynamics = {}
        for t_id, queue in trajectories.items():
            pts = list(queue)
            if len(pts) < 3:
                # Filter out trees/plants that are too new to have danger score
                if class_names is not None:
                    cls_name = class_names[pts[-1]['class']].lower()
                    if "tree" in cls_name or "plant" in cls_name:
                        continue
                
                dynamics[t_id] = {
                    'v': 0, 'a': 0, 'ttc': 999, 'inv_ttc': 0, 
                    'threat_score': 0, 'y_max': pts[-1]['bbox'][3] if len(pts) > 0 else 0, 
                    'bbox': pts[-1]['bbox'], 'class': pts[-1]['class'], 'curvature': 0
                }
                continue

            # Kinematics
            p0, p1, p2 = pts[-3], pts[-2], pts[-1]
            
            # Use 3D velocity if available, else 2D pixel velocity
            if 'bev' in p0 and 'bev' in p2 and p0['bev'][1] != 0:
                v1 = math.hypot(p1['bev'][0] - p0['bev'][0], p1['bev'][1] - p0['bev'][1])
                v2 = math.hypot(p2['bev'][0] - p1['bev'][0], p2['bev'][1] - p1['bev'][1])
            else:
                v1 = math.hypot(p1['cx'] - p0['cx'], p1['cy'] - p0['cy'])
                v2 = math.hypot(p2['cx'] - p1['cx'], p2['cy'] - p1['cy'])
            accel = v2 - v1
            
            # --- Safety Features ---
            growth, ttc = self._calc_looming(p0, p2)
            vx, is_converging_lat = self._calc_lateral(p1, p2)
            
            y_max = p2['bbox'][3]
            proximity = max(0, (y_max - 100) / 380)

            # Ego-Zone Analysis (70% FOV)
            in_ego_path = not (p2['bbox'][2] < 96 or p2['bbox'][0] > 544)
            overlap_bonus = 0.5 if in_ego_path else 0.0

            # Off-Road Detection
            is_off_road = p2['cx'] < 64 or p2['cx'] > 576
            drifting_out = (p2['cx'] < 64 and vx < 0) or (p2['cx'] > 576 and vx > 0)
            road_risk = 0.4 if (is_off_road and drifting_out) else 0.0

            # --- Multivariate Threat Index ---
            base_threat = (0.35 * proximity) + \
                          (0.35 * (1.0/max(1, ttc/15))) + \
                          (0.3 * (min(1.0, abs(vx)/2.0) + (overlap_bonus * proximity) + road_risk))
            
            # Temporal Smoothing (EMA)
            prev_t = pts[-2].get('threat_score', base_threat) if 'threat_score' in pts[-2] else base_threat
            threat_smoothed = min(1.0, 0.7 * base_threat + 0.3 * prev_t)

            # Tree-specific filtering: only consider if in center (ego path) and in danger
            if class_names is not None:
                cls_name = class_names[p2['class']].lower()
                if "tree" in cls_name or "plant" in cls_name:
                    is_danger = threat_smoothed > 0.45
                    if not (in_ego_path and is_danger):
                        continue

            dynamics[t_id] = {
                'v': v2, 'a': accel, 'vx': vx, 'ttc': ttc, 'inv_ttc': 1.0/ttc if ttc > 0 else 0,
                'threat_score': threat_smoothed, 
                'is_converging': (growth > 0.1 and is_converging_lat),
                'off_road_risk': is_off_road and drifting_out, 
                'y_max': y_max,
                'bbox': p2['bbox'], 'class': p2['class'], 
                'curvature': abs(p2['cx'] - p1['cx']) / max(1.0, v2)
            }
            p2['threat_score'] = threat_smoothed # Persist for EMA
            
        return dynamics
