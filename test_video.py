import cv2
import yaml
import glob
from core.perception import PerceptionLayer

def check():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    p = PerceptionLayer(config)
    
    for video_file in glob.glob("data/*.mp4"):
        cap = cv2.VideoCapture(video_file)
        classes_seen = set()
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret or count > 300: break
            count += 1
            h, _ = frame.shape[:2]
            top_crop = int(h * config["perception"]["roi"]["crop_top_percent"] / 100)
            bottom_crop = int(h * config["perception"]["roi"]["crop_bottom_percent"] / 100)
            frame = frame[top_crop:h-bottom_crop, :]
            objs = p.infer_and_track(frame)
            for obj in objs:
                classes_seen.add(p.model.names[obj['class']])
        print(f"Classes in {video_file}:", classes_seen)

if __name__ == "__main__":
    check()
