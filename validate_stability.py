import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os

def generate_synthetic_data(num_frames=1000):
    """
    Mocking the bounding box outputs of a perception model across frames.
    We generate a base trajectory and inject instability events.
    """
    np.random.seed(42)
    time = np.arange(num_frames)
    
    # Base smooth trajectory
    # x represents lateral position in pixels
    # y represents longitudinal position (or bounding box scale/distance)
    base_x = 500 + 50 * np.sin(time / 50.0) 
    base_y = 600 + 20 * np.cos(time / 50.0)
    
    # Introduce instability events to simulate bad driving behavior
    event1_start, event1_end = 200, 250 # Swerving (lateral)
    event2_start, event2_end = 600, 630 # Sudden braking (longitudinal)
    event3_start, event3_end = 800, 850 # High speed lane change / erratic movement
    
    x = base_x.copy()
    y = base_y.copy()
    
    # 1. Swerving (lateral variation)
    x[event1_start:event1_end] += 100 * np.sin(np.arange(50) / 5.0)
    
    # 2. Sudden braking (longitudinal rapid change)
    y[event2_start:event2_end] -= 50 * np.arange(30)
    
    # 3. High variance erratic movement
    x[event3_start:event3_end] += 150 * np.sin(np.arange(50) / 3.0)
    y[event3_start:event3_end] += 30 * np.sin(np.arange(50) / 2.0)
    
    # Add some natural tracker noise
    x += np.random.normal(0, 2, num_frames)
    y += np.random.normal(0, 2, num_frames)
    
    # Ground truth proxy stability indicators (e.g., IMU lateral accel, steering variation)
    # They should naturally correlate with the motion features for validation
    lane_deviation_gt = np.abs(x - 500) / 200.0
    turning_sharpness_gt = np.abs(np.gradient(np.gradient(x))) / 5.0
    trajectory_variance_gt = pd.Series(x).rolling(20, min_periods=1).var().fillna(0).values / 1000.0
    
    # Combine to represent an integrated vehicle instability ground truth proxy
    true_instability = (lane_deviation_gt * 0.3 + 
                        turning_sharpness_gt * 0.5 + 
                        trajectory_variance_gt * 0.2)
    
    return pd.DataFrame({
        'frame_id': time,
        'bbox_x': x,
        'bbox_y': y,
        'gt_lane_deviation': lane_deviation_gt,
        'gt_turning_sharpness': turning_sharpness_gt,
        'gt_trajectory_variance': trajectory_variance_gt,
        'gt_instability_proxy': true_instability
    })

def extract_trajectory_features(df):
    """
    Extract relevant motion features from vision-based trajectory.
    1. Lateral deviation
    2. Trajectory curvature
    3. Speed change across frames
    4. Sudden motion variation (jerk)
    """
    # 1. Lateral deviation (from a smoothed moving average path)
    smoothed_x = df['bbox_x'].rolling(window=30, min_periods=1).mean()
    df['lateral_deviation'] = np.abs(df['bbox_x'] - smoothed_x)
    
    # 2. Trajectory curvature (using derivatives of position)
    dx = np.gradient(df['bbox_x'])
    dy = np.gradient(df['bbox_y'])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature formula: kappa = |dx*ddy - dy*ddx| / (dx^2 + dy^2)^(3/2)
    denominator = (dx**2 + dy**2)**1.5 + 1e-6
    df['trajectory_curvature'] = np.abs(dx * ddy - dy * ddx) / denominator
    # Cap extremely high curvatures when speed is near zero
    df['trajectory_curvature'] = np.clip(df['trajectory_curvature'], 0, 1.0)
    
    # 3. Speed change across frames (acceleration magnitude)
    speed = np.sqrt(dx**2 + dy**2)
    df['speed_change'] = np.abs(np.gradient(speed))
    
    # 4. Sudden motion variation (jerk - derivative of acceleration)
    df['motion_variation'] = np.abs(np.gradient(df['speed_change']))
    
    return df

def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def compute_dss(df):
    """
    Compute Dynamic Stability Score (DSS). Higher score means higher instability.
    Combines normalized trajectory features.
    """
    lat_dev = min_max_scale(df['lateral_deviation'])
    curv = min_max_scale(df['trajectory_curvature'])
    accel = min_max_scale(df['speed_change'])
    jerk = min_max_scale(df['motion_variation'])
    
    # DSS formulation using weighted summation
    w_lat, w_curv, w_acc, w_jerk = 0.3, 0.2, 0.2, 0.3
    df['dss'] = w_lat * lat_dev + w_curv * curv + w_acc * accel + w_jerk * jerk
    
    # Smooth DSS slightly for temporal stability in video
    df['dss_smoothed'] = df['dss'].rolling(window=10, min_periods=1).mean()
    
    return df

def evaluate_and_plot(df, output_dir='.'):
    print("=== Correlation Analysis ===")
    proxies = ['gt_lane_deviation', 'gt_turning_sharpness', 'gt_trajectory_variance', 'gt_instability_proxy']
    
    for proxy in proxies:
        pearson_corr, p_p = pearsonr(df['dss_smoothed'], df[proxy])
        spearman_corr, p_s = spearmanr(df['dss_smoothed'], df[proxy])
        print(f"DSS vs {proxy}:")
        print(f"  Pearson:  {pearson_corr:.3f} (p={p_p:.2e})")
        print(f"  Spearman: {spearman_corr:.3f} (p={p_s:.2e})")
    print("==============================\n")
        
    # Visualizations
    sns.set_theme(style="whitegrid")
    
    # 1. Trajectory deviation vs stability score
    plt.figure(figsize=(10, 5))
    scatter = plt.scatter(df['lateral_deviation'], df['dss_smoothed'], alpha=0.6, 
                          c=df['dss_smoothed'], cmap='coolwarm', edgecolor='k', s=40)
    plt.title('Vision Trajectory Deviation vs. Dynamic Stability Score (DSS)')
    plt.xlabel('Lateral Deviation (Pixels)')
    plt.ylabel('Dynamic Stability Score (DSS)')
    plt.colorbar(scatter, label='Instability Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dss_vs_deviation.png'), dpi=300)
    plt.close()
    
    # 2. DSS over time compared to ground truth proxy
    plt.figure(figsize=(12, 5))
    plt.plot(df['frame_id'], df['dss_smoothed'], label='Predicted DSS (Vision-Based)', color='red', linewidth=2)
    plt.plot(df['frame_id'], min_max_scale(df['gt_instability_proxy']) * df['dss_smoothed'].max(), 
             label='Ground Truth Proxy (Scaled)', color='blue', alpha=0.6, linestyle='--')
    
    # Highlight instability events injected in the synthetic data
    plt.axvspan(200, 250, color='orange', alpha=0.2, label='Event: Swerving')
    plt.axvspan(600, 630, color='purple', alpha=0.2, label='Event: Sudden Braking')
    plt.axvspan(800, 850, color='black', alpha=0.2, label='Event: High-Speed Erratic Change')
    
    plt.title('Dynamic Stability Score (DSS) Over Time')
    plt.xlabel('Video Frame ID')
    plt.ylabel('Instability Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dss_over_time.png'), dpi=300)
    plt.close()
    
    # 3. Instability events vs trajectory changes
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    axs[0].plot(df['frame_id'], df['bbox_x'], color='green')
    axs[0].set_ylabel('BBox X (Lateral Pos)')
    axs[0].set_title('Vision Tracked Object Trajectory')
    
    axs[1].plot(df['frame_id'], df['speed_change'], color='orange')
    axs[1].set_ylabel('Frame-to-Frame Speed Change')
    axs[1].set_title('Extracted Motion Acceleration Feature')
    
    axs[2].plot(df['frame_id'], df['dss_smoothed'], color='red')
    axs[2].set_ylabel('DSS')
    axs[2].set_xlabel('Video Frame ID')
    axs[2].set_title('Resulting Dynamic Stability Score')
    
    # Add spans for events on all subplots
    for ax in axs:
        ax.axvspan(200, 250, color='gray', alpha=0.2)
        ax.axvspan(600, 630, color='gray', alpha=0.2)
        ax.axvspan(800, 850, color='gray', alpha=0.2)
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_changes_events.png'), dpi=300)
    plt.close()
    
    print("Files successfully generated:")
    print(" - dss_vs_deviation.png")
    print(" - dss_over_time.png")
    print(" - trajectory_changes_events.png")

if __name__ == "__main__":
    print("Generating vision-based trajectory mock data...")
    df_data = generate_synthetic_data(1000)
    
    print("Extracting trajectory features...")
    df_features = extract_trajectory_features(df_data)
    
    print("Computing Dynamic Stability Score (DSS)...")
    df_scored = compute_dss(df_features)
    
    print("Plotting and correlating validation pipeline...")
    evaluate_and_plot(df_scored)
