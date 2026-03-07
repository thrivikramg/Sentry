import cv2
import numpy as np

class Visualizer:
    def __init__(self, config):
        """Initializes the visualizer with configuration."""
        self.config = config
        self.risky_ids = set() # Track IDs currently in a Red state
        
        # EMA filters for HUD stability
        self.ema_fps = 0.0
        self.ema_agg = 0.0
        self.ema_idx = 0.0
        self.alpha_hud = 0.2  # Smoothing factor (Lower = smoother)

    def draw(self, frame, trajectories, dynamics, agg_metric, event_state, dss, stability_idx, policy, fps, collision_risk=False, evasion_dir="NONE", class_names=None, zone="UNKNOWN", zone_risk=0.0, fed_threshold=0.7):
        """
        Overlays premium graphical components for a state-of-the-art research demonstration.
        """
        viz = frame.copy()
        _ = viz.shape[:2]
        
        # 1. Cleanup lost risky IDs
        current_frame_ids = set(dynamics.keys())
        self.risky_ids = self.risky_ids.intersection(current_frame_ids)

        # 2. Premium Bounding Boxes (Corner Accents)
        for t_id, dyn in dynamics.items():
            bbox = dyn['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Risk Latching logic (Hysteresis)
            # Entry threshold: 0.45 (More sensitive for side cut-ins)
            # Exit threshold: 0.15 (Sticky Red)
            current_threat = dyn['threat_score']
            if t_id not in self.risky_ids:
                if current_threat > 0.45:
                    self.risky_ids.add(t_id)
            else:
                if current_threat < 0.15:
                    self.risky_ids.remove(t_id)

            is_risk = t_id in self.risky_ids
            color = (0, 0, 255) if is_risk else (0, 255, 0) # BGR
            
            # Semi-transparent box fill (subtle)
            overlay = viz.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.1, viz, 0.9, 0, viz)
            
            # Corner accents instead of full rectangle
            length = min(20, (x2-x1)//4, (y2-y1)//4)
            # Top Left
            cv2.line(viz, (x1, y1), (x1+length, y1), color, 2)
            cv2.line(viz, (x1, y1), (x1, y1+length), color, 2)
            # Top Right
            cv2.line(viz, (x2, y1), (x2-length, y1), color, 2)
            cv2.line(viz, (x2, y1), (x2, y1+length), color, 2)
            # Bottom Left
            cv2.line(viz, (x1, y2), (x1+length, y2), color, 2)
            cv2.line(viz, (x1, y2), (x1, y2-length), color, 2)
            # Bottom Right
            cv2.line(viz, (x2, y2), (x2-length, y2), color, 2)
            cv2.line(viz, (x2, y2), (x2, y2-length), color, 2)
            
            # Scitech Label
            cls_name = class_names[dyn['class']].upper() if class_names and 'class' in dyn else "OBJ"
            
            # Extract historical BEV Z data to show 3D distance
            z_dist_str = ""
            if t_id in trajectories and len(trajectories[t_id]) > 0:
                last_pt = trajectories[t_id][-1]
                if 'bev' in last_pt and last_pt['bev'][1] != 0:
                    z = last_pt['bev'][1]
                    z_dist_str = f" | z:{z:.1f}m"
            
            label = f"{cls_name}_{t_id} | v:{dyn['v']:.1f}{z_dist_str}"
            cv2.putText(viz, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw Tail (Trajectory)
            if t_id in trajectories:
                pts = list(trajectories[t_id])
                for i in range(1, len(pts)):
                    p1 = (int(pts[i-1]['cx']), int(pts[i-1]['cy']))
                    p2 = (int(pts[i]['cx']), int(pts[i]['cy']))
                    cv2.line(viz, p1, p2, color, 1)
                    
        # 2. Futuristic HUD
        # Apply smoothing to live metrics for UI stability
        self.ema_fps = (self.alpha_hud * fps) + (1 - self.alpha_hud) * self.ema_fps
        self.ema_agg = (self.alpha_hud * agg_metric) + (1 - self.alpha_hud) * self.ema_agg
        self.ema_idx = (self.alpha_hud * stability_idx) + (1 - self.alpha_hud) * self.ema_idx

        # Top Left Overlay
        hud_overlay = viz.copy()
        cv2.rectangle(hud_overlay, (0, 0), (350, 260), (20, 20, 20), -1)
        cv2.addWeighted(hud_overlay, 0.7, viz, 0.3, 0, viz)
        
        # Grid lines for HUD feel
        for i in range(0, 260, 40):
            cv2.line(viz, (0, i), (350, i), (50, 50, 50), 1)
        
        # Data Readouts
        cv2.putText(viz, "DSS SYSTEM CORE v1.0", (15, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(viz, f"TELEMETRY FPS: {self.ema_fps:.1f}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Federated Overlays
        cv2.putText(viz, f"GEO-ZONE: {zone}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 255), 1)
        cv2.putText(viz, f"CLOUD RISK MAP: {zone_risk:.2f} | FED THRESH: {fed_threshold:.2f}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 100), 1)

        # Scene State
        state_color = (0, 0, 255) if event_state in ["aggressive", "critical", "crash"] else (0, 255, 0)
        cv2.putText(viz, f"SCENE STATE: {event_state.upper()}", (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        cv2.putText(viz, f"AGGRESSION INDEX: {self.ema_agg:.3f}", (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # DSS Score Gauge
        cv2.putText(viz, "STABILITY CONFIDENCE", (15, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        gauge_w = 200
        fill_w = int(gauge_w * self.ema_idx)
        cv2.rectangle(viz, (15, 190), (15 + gauge_w, 200), (100, 100, 100), 1)
        
        # Gradient bar
        if self.ema_idx > 0.7:
            bar_color = (0, 255, 0)
        elif self.ema_idx > 0.3:
            bar_color = (0, 255, 255)
        else:
            bar_color = (0, 0, 255)
        cv2.rectangle(viz, (15, 190), (15 + fill_w, 200), bar_color, -1)
        cv2.putText(viz, f"{dss:.1f}", (15 + gauge_w + 10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Policy Action
        cv2.rectangle(viz, (15, 220), (305, 245), (40, 40, 40), -1)
        cv2.putText(viz, f"POLICY: {policy}", (22, 238), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

        if collision_risk:
            # Huge flashing banner in the middle
            h_v, w_v = viz.shape[:2]
            overlay = viz.copy()
            
            # Change banner for active crash vs looming
            if event_state == "crash":
                cv2.rectangle(overlay, (0, h_v//2 - 60), (w_v, h_v//2 + 60), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.9, viz, 0.1, 0, viz)
                cv2.putText(viz, "CRASH DETECTED", (w_v//2 - 200, h_v//2 + 15), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3)
            else:
                cv2.rectangle(overlay, (w_v//2 - 220, h_v//2 - 40), (w_v//2 + 220, h_v//2 + 40), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.8, viz, 0.2, 0, viz)
                cv2.putText(viz, "COLLISION IMMINENT", (w_v//2 - 180, h_v//2 + 10), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(viz, f"ACTION: {evasion_dir}", (w_v//2 - 120, h_v//2 + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return viz
