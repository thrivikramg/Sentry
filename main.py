import cv2
import time
import yaml
import numpy as np
from collections import deque

# Assume modular imports (we will build these later)
from core.perception import PerceptionLayer
from core.tracking import TrackingLayer
from core.analysis import TrajectoryAnalyzer
from core.stability import StabilityEngine
from core.federation import MockFederatedServer, FederatedClient
from utils.visualization import Visualizer
from utils.evaluation import EvaluationLogger
from generate_phd_plots import generate_academic_plots

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline(video_source, config_path="config.yaml"):
    config = load_config(config_path)
    
    # Initialize Modules
    perception = PerceptionLayer(config)
    tracker = TrackingLayer(config)
    analysis = TrajectoryAnalyzer(config)
    stability = StabilityEngine(config)
    baseline_stability = StabilityEngine(config) # Unconnected baseline tracking
    visualizer = Visualizer(config)
    evaluator = EvaluationLogger(config)

    # Initialize Federated Network Simulation
    fed_server = MockFederatedServer()
    # Pre-pollute ZONE_B to simulate historical crash data on the risk map
    fed_server.report_risk("ZONE_B (Downtown Hotspot)", 1.5, True) 
    fed_server.report_risk("ZONE_B (Downtown Hotspot)", 1.2, True) 
    fed_client = FederatedClient(client_id="EGO_VEHICLE_01", server=fed_server, base_config=config)

    # Input Stream
    if str(video_source).isdigit():
        video_source = int(video_source)
    cap = cv2.VideoCapture(video_source)
    
    # FPS Tracking
    frame_times = deque(maxlen=30)
    
    print("--- DSS System Starting ---")
    
    frame_count = 0
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
            
        frame_count += 1
        
        # FOV Cropping (Optimization for hood/dashboard removal)
        h, _ = frame.shape[:2]
        top_crop = int(h * config["perception"]["roi"]["crop_top_percent"] / 100)
        bottom_crop = int(h * config["perception"]["roi"]["crop_bottom_percent"] / 100)
        frame = frame[top_crop:h-bottom_crop, :]

        # 1. Perception & Base Tracking (YOLOv8 + ByteTrack)
        tracked_objects = perception.infer_and_track(frame)
        
        # 2. Trajectory History Management
        trajectories = tracker.update(tracked_objects)
        
        # 3. Trajectory Analysis (Kinematics & TTC)
        dynamics = analysis.compute_dynamics(trajectories, class_names=perception.model.names)
        
        # 4. Behavioral Feature Engine & DSS Calculation
        # Calculate baseline behavior (unconnected car)
        _, baseline_event_state, _, _ = baseline_stability.compute_aggression(dynamics)
        baseline_dss, _, _ = baseline_stability.update_dss(baseline_event_state)
        
        # Calculate federated behavior (connected car)
        agg_metric, event_state, collision_risk, evasion_dir = stability.compute_aggression(dynamics)
        dss, stability_idx, policy = stability.update_dss(event_state)
        
        # --- FEDERATED RISK MAPPING: Telemetry & Parameter Sync ---
        fed_client.update_zone(frame_count)
        fed_client.send_telemetry(agg_metric, collision_risk)
        updated_params, zone_risk = fed_client.sync_config()
        
        # Dynamically mutate the vehicle's safe driving threshold based on cloud knowledge!
        stability.aggression_threshold = updated_params['behavioral_engine']['aggression_threshold']

        # 5. Evaluation Logging
        evaluator.log_frame(frame_count, dss, event_state, policy, agg_metric, baseline_event_state, baseline_dss)
        
        # FPS Calc
        end_time = time.time()
        frame_times.append(end_time - start_time)
        fps = 1.0 / (sum(frame_times) / len(frame_times))

        # 6. Visualization
        if config["ui"]["show_video"]:
            viz_frame = visualizer.draw(
                frame, trajectories, dynamics, 
                agg_metric, event_state, 
                dss, stability_idx, policy, fps,
                collision_risk, evasion_dir,
                class_names=perception.model.names,
                zone=fed_client.current_zone,
                zone_risk=zone_risk,
                fed_threshold=stability.aggression_threshold
            )
            cv2.imshow("Dynamic Stability Scoring (DSS)", viz_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    evaluator.save_report("data/evaluation_report.csv")
    evaluator.save_plots("data/plots")
    generate_academic_plots("data/plots")
    print("--- DSS System Shutdown ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run DSS Framework")
    parser.add_argument("--source", type=str, default="0", help="Video file path or webcam index (0)")
    args = parser.parse_args()
    
    run_pipeline(args.source)
