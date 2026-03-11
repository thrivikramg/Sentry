import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def extract_trajectory_features(df):
    """
    Extract relevant motion features from vision-based trajectory.
    """
    if len(df) < 30:
        smoothed_x = df['bbox_x'].rolling(window=5, min_periods=1).mean()
    else:
        smoothed_x = df['bbox_x'].rolling(window=30, min_periods=1).mean()
        
    df['lateral_deviation'] = np.abs(df['bbox_x'] - smoothed_x)
    
    # Trajectory curvature
    dx = np.gradient(df['bbox_x'])
    dy = np.gradient(df['bbox_y'])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    denominator = (dx**2 + dy**2)**1.5 + 1e-6
    df['trajectory_curvature'] = np.abs(dx * ddy - dy * ddx) / denominator
    df['trajectory_curvature'] = np.clip(df['trajectory_curvature'], 0, 1.0)
    
    # Speed change across frames
    speed = np.sqrt(dx**2 + dy**2)
    df['speed_change'] = np.abs(np.gradient(speed))
    
    # Sudden motion variation (jerk)
    df['motion_variation'] = np.abs(np.gradient(df['speed_change']))
    
    return df

def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def compute_dss(df):
    """
    Compute Dynamic Stability Score (DSS). Higher score means higher instability.
    """
    lat_dev = min_max_scale(df['lateral_deviation'])
    curv = min_max_scale(df['trajectory_curvature'])
    accel = min_max_scale(df['speed_change'])
    jerk = min_max_scale(df['motion_variation'])
    
    w_lat, w_curv, w_acc, w_jerk = 0.3, 0.2, 0.2, 0.3
    df['dss'] = w_lat * lat_dev + w_curv * curv + w_acc * accel + w_jerk * jerk
    df['dss_smoothed'] = df['dss'].rolling(window=10, min_periods=1).mean()
    
    return df

def evaluate_and_plot(df, output_dir='.'):
    sns.set_theme(style="whitegrid")
    
    # 1. Trajectory deviation vs stability score
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(df['lateral_deviation'], df['dss_smoothed'], alpha=0.6, 
                          c=df['dss_smoothed'], cmap='coolwarm', edgecolor='k', s=40)
    plt.title('KCF Tracked Deviations vs. Dynamic Stability Score (DSS)')
    plt.xlabel('Lateral Deviation (Pixels)')
    plt.ylabel('Dynamic Stability Score (DSS)')
    plt.colorbar(scatter, label='Instability Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kcf_dss_vs_deviation.png'), dpi=300)
    plt.close()
    
    # 2. DSS over time
    plt.figure(figsize=(12, 5))
    plt.plot(df['frame_id'], df['dss_smoothed'], label='Predicted DSS (KCF-Based)', color='red', linewidth=2)
    plt.title('Dynamic Stability Score (DSS) Over Time (KCF Tracker)')
    plt.xlabel('Video Frame ID')
    plt.ylabel('Instability Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kcf_dss_over_time.png'), dpi=300)
    plt.close()
    
    # 3. Trajectory vs Speed Change vs DSS
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axs[0].plot(df['frame_id'], df['bbox_x'], color='green')
    axs[0].set_ylabel('BBox X (Lateral Pos)')
    axs[0].set_title('KCF Tracked Object Trajectory')
    
    axs[1].plot(df['frame_id'], df['speed_change'], color='orange')
    axs[1].set_ylabel('Frame-to-Frame Speed Change')
    axs[1].set_title('Extracted Motion Acceleration Feature')
    
    axs[2].plot(df['frame_id'], df['dss_smoothed'], color='red')
    axs[2].set_ylabel('DSS')
    axs[2].set_xlabel('Video Frame ID')
    axs[2].set_title('Resulting Dynamic Stability Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kcf_trajectory_changes.png'), dpi=300)
    plt.close()
    
    print("Graphs successfully mapped from KCF Tracker:")
    print(" - kcf_dss_vs_deviation.png")
    print(" - kcf_dss_over_time.png")
    print(" - kcf_trajectory_changes.png")


def run_kcf_tracker(video_path):
    print(f"Loading video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video -> {video_path}")
        return None
        
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read the first frame from the video.")
        return None
        
    # Resize frame for proper view
    scale_percent = 50 
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    print("\n[INSTRUCTIONS]")
    print("1. A window will open showing the first frame.")
    print("2. Draw a bounding box around the vehicle you want to track using your mouse.")
    print("3. Press ENTER or SPACE to confirm the box and start KCF tracking.")
    print("4. Press 'Q' at any time to stop tracking and evaluate early.\n")
    
    bbox = cv2.selectROI("Select Vehicle to Track", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Vehicle to Track")
    
    if bbox == (0, 0, 0, 0):
        print("No bounding box selected. Exiting.")
        return None
        
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, bbox)
    
    frame_id = 0
    tracking_data = []

    print("\nTracking started... Press 'q' on the video window to stop.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        success, bbox = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Use the center of the bounding box
            center_x = x + w / 2
            center_y = y + h / 2
            
            tracking_data.append({
                'frame_id': frame_id,
                'bbox_x': center_x,
                'bbox_y': center_y
            })
            
            cv2.putText(frame, "KCF Tracker - Tracking", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "KCF Tracker - LOST TARGET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow("KCF Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
    
    if not tracking_data:
        print("No tracking data captured.")
        return None
        
    return pd.DataFrame(tracking_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KCF Tracker validation for vehicle stability.")
    parser.add_argument("--video", type=str, default="data/test.mp4", help="Path to the video file.")
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"File not found: {args.video}. Please run with `--video path/to/video.mp4`.")
    else:
        df_data = run_kcf_tracker(args.video)
        
        if df_data is not None:
            print(f"\nSuccessfully tracked {len(df_data)} frames.")
            print("Extracting trajectory features...")
            df_features = extract_trajectory_features(df_data)
            
            print("Computing Dynamic Stability Score (DSS)...")
            df_scored = compute_dss(df_features)
            
            print("Generating Plots...")
            evaluate_and_plot(df_scored)
