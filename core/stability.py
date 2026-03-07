import math
import numpy as np

class StabilityEngine:
    def __init__(self, config):
        c = config['behavioral_engine']
        self.w = c['weights']
        self.aggression_threshold = c['aggression_threshold']
        
        d = config['dynamic_stability_score']
        self.lam = d['decay_factor_lambda']
        self.score = d['initial_score']
        self.min_score, self.max_score = d['score_min'], d['score_max']
        self.r_stable, self.r_aggress = d['stable_reward'], d['aggressive_penalty']
        
        self.fps = config['system']['target_fps']
        self.collision_thresh = c['collision_warning_seconds'] * self.fps

    def _normalize(self, value, min_v, max_v):
        if max_v - min_v == 0: return 0
        return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))

    def _get_evasion(self, dynamics, collision, crash):
        """Determines steering and braking response."""
        direction = "NONE"
        brake = False
        if not (collision or crash): return direction, brake

        tid = max(dynamics, key=lambda k: dynamics[k]['threat_score'])
        t = dynamics[tid]
        cx = t['bbox'][0] + (t['bbox'][2] - t['bbox'][0])/2
        
        if t['threat_score'] > 0.75 or t.get('is_converging', False) or crash:
            brake = True
        direction = "STEER RIGHT" if cx < 320 else "STEER LEFT"
        
        if brake:
            direction = f"BRAKE + {direction}" if direction != "NONE" else "EMERGENCY BRAKE"
        return direction, brake

    def compute_aggression(self, dynamics):
        """Computes scene-wide Aggression A(t) and safety alerts."""
        if not dynamics: return 0.0, "stable", False, "NONE"

        max_agg, collision, crash = 0.0, False, False
        for _, d in dynamics.items():
            agg = (self.w['w1'] * self._normalize(d['v'], 0, 50)) + \
                  (self.w['w2'] * self._normalize(d['a'], 0, 20)) + \
                  (self.w['w3'] * self._normalize(d['inv_ttc'], 0, 0.5)) + \
                  (self.w['w4'] * self._normalize(d['curvature'], 0, 0.5))
            
            combined = (agg * 0.4) + (d['threat_score'] * 0.6)
            max_agg = max(max_agg, combined)
            if 0 < d['ttc'] < self.collision_thresh: collision = True
            if d['y_max'] > 440 and d['threat_score'] > 0.85: crash = True

        event = "stable"
        if max_agg > self.aggression_threshold: event = "aggressive"
        if collision: event = "critical"
        if crash: event = "crash"

        evasion_dir, _ = self._get_evasion(dynamics, collision, crash)
        
        # Off-Road handling
        off_road = any(d.get('off_road_risk', False) for d in dynamics.values())
        if off_road and not (collision or crash):
            event, evasion_dir = "critical", "RECOVER TO CENTER"

        return max_agg, event, (collision or crash or off_road), evasion_dir

    def update_dss(self, event_state):
        r_t = self.r_stable if event_state == "stable" else self.r_aggress
        self.score = max(self.min_score, min(self.max_score, (self.lam * self.score) + r_t))
        
        # Sigmoid normalization
        idx = 1 / (1 + math.exp(-(self.score - 50.0) / 10.0))
        
        if idx < 0.2: policy = "CRITICAL: Safety Mode"
        elif idx < 0.4: policy = "LOW: Conservative Behavior"
        elif idx < 0.7: policy = "MEDIUM: Smooth Acceleration"
        else: policy = "HIGH: Normal Speed"
            
        return self.score, idx, policy
