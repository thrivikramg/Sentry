import csv
import matplotlib.pyplot as plt
import os
import numpy as np

class EvaluationLogger:
    def __init__(self, config):
        self.logs = []
        self.spikes = 0
        self.aggressive_events = 0
        self.prev_state = "stable"

    def log_frame(self, frame_idx, dss, state, policy, max_threat=0.0, baseline_state="stable", baseline_dss=0.0):
        """
        Records the system output frame-by-frame.
        """
        # Detect instability spike! 
        if state == "aggressive" and self.prev_state == "stable":
            self.spikes += 1
            
        if state == "aggressive":
            self.aggressive_events += 1
            
        self.prev_state = state
        
        self.logs.append({
            'frame': frame_idx,
            'dss_score': dss,
            'max_threat': max_threat,
            'state': state,
            'policy': policy,
            'baseline_state': baseline_state,
            'baseline_dss_score': baseline_dss
        })

    def save_report(self, filepath):
        """
        Dumps the accumulated frame logs and a summary to disk.
        """
        if not self.logs:
            return
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        print(f"Saving Evaluation Report to {filepath}...")
        
        # We can dump frame data to CSV
        keys = self.logs[0].keys()
        
        with open(filepath, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.logs)
            
        # Summary
        frames = len(self.logs)
        minutes = frames / 30.0 / 60.0 if frames > 0 else 0 
        epm = self.aggressive_events / minutes if minutes > 0 else 0
        
        # Federated Advantage Calculation
        reaction_advantage_frames = 0
        for log in self.logs:
            if log['state'] in ['aggressive', 'critical', 'crash'] and log['baseline_state'] == 'stable':
                reaction_advantage_frames += 1
                
        # Approx seconds saved assuming 30fps
        advantage_seconds = reaction_advantage_frames / 30.0
        
        print("\n=== SYSTEM EVALUATION SUMMARY ===")
        print(f"Total Frames Processed : {frames}")
        print(f"Total Instability Spikes: {self.spikes}")
        print(f"Aggressive Event rate  : {epm:.2f} events/min")
        print(f"Federated Reaction Advantage (Frames): {reaction_advantage_frames} frames early detection!")
        print(f"Federated Reaction Advantage (Seconds): {advantage_seconds:.2f} seconds saved over unconnected cars!")
        print("=================================\n")

    def save_plots(self, folder="data/plots"):
        """
        Generates and saves visual graphs of the drive stability.
        """
        if not self.logs:
            return
            
        os.makedirs(folder, exist_ok=True)
        frames = [log['frame'] for log in self.logs]
        dss_scores = [log['dss_score'] for log in self.logs]
        
        # 1. Stability Plot with Threat Overlay
        _, ax1 = plt.subplots(figsize=(12, 6))
        
        # Stability Score (Left Axis)
        ax1.plot(frames, dss_scores, color='#e74c3c', linewidth=2, label='Connected DSS Score (Federated)')
        if 'baseline_dss_score' in self.logs[0]:
            baseline_dss_scores = [log['baseline_dss_score'] for log in self.logs]
            ax1.plot(frames, baseline_dss_scores, color='#2c3e50', linewidth=2, linestyle='--', label='Unconnected DSS Score (Baseline)')
            
        ax1.axhline(y=30, color='red', linestyle=':', alpha=0.5, label='Critical Hazard Threshold')
        ax1.set_xlabel("Frame Time (idx)")
        ax1.set_ylabel("Stability Index (Score)", color='#2c3e50', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#2c3e50')
        ax1.set_ylim(0, 105)
        
        # Max Threat Score (Right Axis)
        ax2 = ax1.twinx()
        threat_scores = [log['max_threat'] for log in self.logs]
        ax2.plot(frames, threat_scores, color='#e67e22', linewidth=1.5, alpha=0.7, label='Peak Object Threat')
        ax2.set_ylabel("Threat Analysis (0-1.0 Scale)", color='#e67e22', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#e67e22')
        ax2.set_ylim(0, 1.1)

        plt.title("Federated DSS vs Baseline Unconnected DSS Scoring Strategy", fontsize=14)
        
        # Unified Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{folder}/stability_threat_correlation.png")
        plt.close()
        
        # 2. State Distribution
        states = [log['state'] for log in self.logs]
        state_counts = {s: states.count(s) for s in set(states)}
        
        plt.figure(figsize=(8, 8))
        plt.pie(state_counts.values(), labels=state_counts.keys(), autopct='%1.1f%%', 
                colors=['#2ecc71', '#e74c3c', '#f1c40f', '#3498db'])
        plt.title("Driving State Distribution")
        plt.savefig(f"{folder}/state_distribution.png")
        plt.close()
        
        # 3. Federated Reaction Delta Plot
        fed_alert = [1 if log['state'] in ['aggressive', 'critical', 'crash'] else 0 for log in self.logs]
        unconnected_alert = [1 if log['baseline_state'] in ['aggressive', 'critical', 'crash'] else 0 for log in self.logs]
        
        plt.figure(figsize=(12, 4))
        plt.plot(frames, fed_alert, label='Connected Car (Federated Config)', color='#e74c3c', linestyle='-', linewidth=2, alpha=0.8)
        plt.plot(frames, unconnected_alert, label='Unconnected Car (Base Config)', color='#34495e', linestyle='--', linewidth=2, alpha=0.8)
        
        plt.fill_between(frames, unconnected_alert, fed_alert, where=(np.array(fed_alert) > np.array(unconnected_alert)),
                         color='#e74c3c', alpha=0.2, label='Reaction Time Advantage (Federated Setup)')
                         
        plt.yticks([0, 1], ['Stable', 'Hazard Alert (Aggressive+)'])
        plt.xlabel("Frame Sequence")
        plt.title("Autonomous Hazard Reaction Time: Connected vs Unconnected Vehicle")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{folder}/federated_reaction_advantage.png")
        plt.close()
        
        print(f"Graphs saved to {folder}/")
