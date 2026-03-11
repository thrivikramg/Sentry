import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_mock_trajectory_data(num_frames=500):
    """
    Generate mock bounding box tracking data with an instability event
    to simulate standard vision-based tracking dataframe.
    """
    np.random.seed(42)
    frame_id = np.arange(num_frames)
    
    # Base straight-line driving (smoothly varying x, linearly increasing y)
    base_x = 500 + 10 * np.sin(frame_id / 20.0)
    base_y = 600 + np.linspace(0, 100, num_frames)
    
    # Inject a sudden swerving event around frame 200 to 250
    event_start, event_end = 200, 250
    base_x[event_start:event_end] += 150 * np.sin(np.arange(50) / 8.0)
    
    # Add mild tracker noise
    bbox_x = base_x + np.random.normal(0, 1.5, num_frames)
    bbox_y = base_y + np.random.normal(0, 1.5, num_frames)
    
    df = pd.DataFrame({
        'frame_id': frame_id,
        'bbox_x': bbox_x,
        'bbox_y': bbox_y,
        'is_actual_event': False
    })
    df.loc[event_start:event_end, 'is_actual_event'] = True
    
    return df

def extract_features(df):
    """
    Compute motion features needed for Dynamic Stability Score (DSS).
    Input strictly matches [frame_id, bbox_x, bbox_y].
    """
    df = df.copy()
    
    # 1. Lateral deviation (vs a trailing 30-frame window path average)
    smoothed_x = df['bbox_x'].rolling(window=30, min_periods=1).mean()
    df['lateral_deviation'] = np.abs(df['bbox_x'] - smoothed_x)
    
    # 2. Kinematics (Gradients)
    dx = np.gradient(df['bbox_x'])
    dy = np.gradient(df['bbox_y'])
    
    # 3. Speed, Speed Change, Acceleration
    df['speed'] = np.sqrt(dx**2 + dy**2)
    df['speed_change'] = np.abs(np.gradient(df['speed']))
    df['acceleration'] = np.gradient(df['speed'])
    
    # 4. Trajectory curvature approximation
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    denominator = (dx**2 + dy**2)**1.5 + 1e-6
    df['curvature'] = np.abs(dx * ddy - dy * ddx) / denominator
    df['curvature'] = np.clip(df['curvature'], 0, 1.0)
    
    return df

def min_max_scale(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-6)

def compute_raw_dss(df):
    """
    Compute Raw Dynamic Stability Score based on standardized features.
    """
    lat_dev = min_max_scale(df['lateral_deviation'])
    curv = min_max_scale(df['curvature'])
    accel = min_max_scale(np.abs(df['acceleration']))
    
    # Weighted calculation
    w_lat, w_curv, w_acc = 0.4, 0.3, 0.3
    raw_dss = w_lat * lat_dev + w_curv * curv + w_acc * accel
    return raw_dss

def add_noise_to_trajectory(df, noise_std=5.0):
    """
    Simulate tracking noise / prediction flicker randomly distributing offset onto boxes.
    """
    noisy_df = df.copy()
    np.random.seed(101)
    noisy_df['bbox_x'] += np.random.normal(0, noise_std, len(noisy_df))
    noisy_df['bbox_y'] += np.random.normal(0, noise_std, len(noisy_df))
    return noisy_df

def evaluate_metrics_and_plot(df_clean, df_noisy, output_dir='.'):
    # --- 1. Processing and Smoothing ---
    df_clean = extract_features(df_clean)
    df_noisy = extract_features(df_noisy)
    
    df_clean['raw_dss'] = compute_raw_dss(df_clean)
    df_noisy['raw_dss'] = compute_raw_dss(df_noisy)
    
    # Temporal smoothing to reduce noise (Exponential Moving Average)
    alpha = 0.2
    df_clean['smoothed_dss'] = df_clean['raw_dss'].ewm(alpha=alpha, adjust=False).mean()
    df_noisy['smoothed_dss'] = df_noisy['raw_dss'].ewm(alpha=alpha, adjust=False).mean()
    
    # --- 2. Temporal Stability Metrics ---
    # Rolling variance of DSS
    window_size = 15
    df_clean['dss_rolling_var'] = df_clean['smoothed_dss'].rolling(window=window_size, min_periods=1).var().fillna(0)
    df_clean['dss_rolling_std'] = df_clean['smoothed_dss'].rolling(window=window_size, min_periods=1).std().fillna(0)
    
    # Temporal consistency score (Inverse of mean absolute frame-to-frame difference)
    frame_diffs = np.abs(np.diff(df_clean['smoothed_dss']))
    temporal_consistency_score = 1.0 / (1.0 + np.mean(frame_diffs))
    
    # Robustness Metrics (Clean vs Noisy)
    correlation = df_clean['smoothed_dss'].corr(df_noisy['smoothed_dss'])
    mean_abs_diff = np.mean(np.abs(df_clean['smoothed_dss'] - df_noisy['smoothed_dss']))
    temporal_stability_ratio = np.mean(df_noisy['smoothed_dss']) / (np.mean(df_clean['smoothed_dss']) + 1e-6)
    
    print("=== Robustness to Prediction Flicker Metrics ===")
    print(f"Correlation (Original vs Noisy DSS): {correlation:.3f}")
    print(f"Mean Abs DSS Difference:             {mean_abs_diff:.3f}")
    print(f"Temporal Stability Ratio:            {temporal_stability_ratio:.3f}")
    print(f"Temporal Consistency Score (Clean):  {temporal_consistency_score:.3f}")
    print("==============================================\n")
    
    # --- 3. Define New Evaluation Metrics ---
    
    # A) Safety Gap
    # Define an instability detection threshold logic dynamically (e.g. mean + 1.5*std)
    threshold = df_clean['smoothed_dss'].mean() + 1.5 * df_clean['smoothed_dss'].std()
    
    # Isolate first frame where DSS exceeds computed threshold
    detected_frames = df_clean[df_clean['smoothed_dss'] > threshold]['frame_id'].values
    first_trigger = detected_frames[0] if len(detected_frames) > 0 else -1
    
    # Frame where physical event actually began
    event_frames = df_clean[df_clean['is_actual_event'] == True]['frame_id'].values
    actual_event_start = event_frames[0] if len(event_frames) > 0 else -1
    
    if first_trigger != -1 and actual_event_start != -1:
        safety_gap = first_trigger - actual_event_start
        print(f"=== Safety Gap Analysis ===")
        print(f"Actual Event Start Frame:       {actual_event_start}")
        print(f"Model First DSS Trigger Frame:  {first_trigger}")
        print(f"Safety Gap (frames):            {safety_gap} frames")
        print(" -> (Positive gap = reaction lag | Negative gap = proactive predictive spike)")
        print("===========================\n")

    # B) Trajectory Instability Score (TIS)
    num_frames = len(df_clean)
    lat_s = min_max_scale(df_clean['lateral_deviation'])
    curv_s = min_max_scale(df_clean['curvature'])
    acc_s = min_max_scale(np.abs(df_clean['acceleration']))
    
    tis = np.sum(acc_s + curv_s + lat_s) / num_frames
    print(f"=== Trajectory Instability Score (TIS) ===")
    print(f"TIS: {tis:.4f}")
    print("==========================================\n")
    
    # --- 4. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Raw vs Smoothed DSS over Video Frames
    plt.figure(figsize=(10, 5))
    plt.plot(df_clean['frame_id'], df_clean['raw_dss'], label='Raw DSS', color='lightcoral', alpha=0.6)
    plt.plot(df_clean['frame_id'], df_clean['smoothed_dss'], label='Smoothed DSS (EMA)', color='darkred', linewidth=2)
    if actual_event_start != -1:
        plt.axvspan(event_frames[0], event_frames[-1], color='orange', alpha=0.2, label='Actual Event Window')
    if first_trigger != -1:
        plt.axvline(first_trigger, color='blue', linestyle='--', label=f'Trigger Threshold Crossed')
    
    plt.title('Raw DSS vs Smoothed DSS over Video Frames')
    plt.xlabel('Frame ID')
    plt.ylabel('Dynamic Stability Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dss_raw_vs_smoothed.png'), dpi=300)
    plt.close()
    
    # Plot 2: DSS Variance over time
    plt.figure(figsize=(10, 5))
    plt.plot(df_clean['frame_id'], df_clean['dss_rolling_var'], color='purple', linewidth=2, label='DSS Rolling Variance (15 frames)')
    if actual_event_start != -1:
        plt.axvspan(event_frames[0], event_frames[-1], color='orange', alpha=0.2, label='Actual Event Window')
    plt.title('Temporal Stability (Rolling Variance) Over Time')
    plt.xlabel('Frame ID')
    plt.ylabel('Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dss_variance_over_time.png'), dpi=300)
    plt.close()
    
    # Plot 3: Comparison of DSS under noise perturbation
    plt.figure(figsize=(10, 5))
    plt.plot(df_clean['frame_id'], df_clean['smoothed_dss'], label='Clean Tracked DSS', color='darkgreen', linewidth=2)
    plt.plot(df_noisy['frame_id'], df_noisy['smoothed_dss'], label='Noisy Tracked DSS (Flicker Simulated)', color='lightgreen', linestyle='dashed', linewidth=2)
    if actual_event_start != -1:
        plt.axvspan(event_frames[0], event_frames[-1], color='orange', alpha=0.2, label='Actual Event Window')
    plt.title('Robustness to Prediction Flicker (Noise Perturbation)')
    plt.xlabel('Frame ID')
    plt.ylabel('Smoothed DSS')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dss_noise_perturbation.png'), dpi=300)
    plt.close()
    
    print("Files successfully generated:")
    print(" - dss_raw_vs_smoothed.png")
    print(" - dss_variance_over_time.png")
    print(" - dss_noise_perturbation.png")

if __name__ == "__main__":
    print("Generating base trajectory dataframe (representing vision tracking)...")
    df_clean = generate_mock_trajectory_data(500)
    
    print("Simulating trajectory with prediction flicker...")
    df_noisy = add_noise_to_trajectory(df_clean, noise_std=5.0)
    
    print("Evaluating temporal stability, computing metrics, and plotting...\n")
    evaluate_metrics_and_plot(df_clean, df_noisy)
