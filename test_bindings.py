import cv2
import yaml
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.perception import PerceptionLayer
from core.tracking import TrackingLayer
from core.analysis import TrajectoryAnalyzer
from core.stability import StabilityEngine
from utils.visualization import Visualizer
from utils.evaluation import EvaluationLogger

def test_pipeline():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    config['ui']['show_video'] = False # Headless mode test
    
    p = PerceptionLayer(config)
    t = TrackingLayer(config)
    a = TrajectoryAnalyzer(config)
    s = StabilityEngine(config)
    v = Visualizer(config)
    e = EvaluationLogger(config)
    
    print("Models Initialized. Testing dummy frame...")
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    track = p.infer_and_track(frame)
    print("Tracked objs: ", len(track))
    
    traj = t.update(track)
    dyn = a.compute_dynamics(traj)
    agg, event = s.compute_aggression(dyn)
    dss, idx, pol = s.update_dss(event)
    
    print("Evaluation Check:", agg, event, dss, idx, pol)
    print("TEST PASSED")

if __name__ == "__main__":
    test_pipeline()
