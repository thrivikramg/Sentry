import torch
import cv2
import numpy as np
from ultralytics import YOLO

class PerceptionLayer:
    def __init__(self, config):
        self.config = config['perception']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading Perception Model: {self.config['model_id']} on {self.device}")
        self.model = YOLO(self.config['model_id'])
        
        self.confidence_thresh = self.config['confidence_threshold']
        self.target_classes = self.config['target_classes']
        self.focal_length = self.config.get('focal_length_pseudo', 500.0)
        
        # Monocular Depth Estimation
        depth_model_name = self.config.get('depth_model', 'MiDaS_small')
        print(f"Loading Depth Model: {depth_model_name}")
        self.midas = torch.hub.load("intel-isl/MiDaS", depth_model_name)
        self.midas.to(self.device)
        self.midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform if depth_model_name == "MiDaS_small" else midas_transforms.default_transform

    def infer_and_track(self, frame):
        """
        Runs YOLOv8 using native ByteTrack.
        Output format: list of dicts [{'id': track_id, 'bbox': [x1,y1,x2,y2], 'class': cls_id}]
        """
        # Monocular Depth Estimation
        input_batch = self.transform(frame).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()

        # ByteTrack natively integrated in ultralytics
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, device=self.device)
        
        tracked_objects = []
        center_x_image = frame.shape[1] / 2.0
        
        if len(results) > 0 and results[0].boxes.id is not None:
            boxes = results[0].boxes
            for idx, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                # Filter by class and confidence
                if conf >= self.confidence_thresh and (not self.target_classes or cls_id in self.target_classes):
                    track_id = int(box.id[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 3D BEV Projection
                    # Sample median disparity from the object's bounding box
                    box_depth_crop = depth_map[int(y1):int(y2), int(x1):int(x2)]
                    if box_depth_crop.size > 0:
                        disp = np.median(box_depth_crop)
                    else:
                        disp = 1.0 # fallback
                    
                    # Inverse Perspective Mapping (IPM) Proxy
                    # MiDaS outputs relative disparity. Z is inversely proportional.
                    Z = 1000.0 / (disp + 1e-6)
                    
                    cx = (x1 + x2) / 2.0
                    X = (cx - center_x_image) * Z / self.focal_length
                    
                    tracked_objects.append({
                        'id': track_id,
                        'bbox': [x1, y1, x2, y2],
                        'class': cls_id,
                        'bev': [X, Z],
                        'disp': disp
                    })
                    
        return tracked_objects
