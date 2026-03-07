import json
import os
import time
from collections import defaultdict

class MockFederatedServer:
    """
    Simulates a Cloud/Edge Aggregation Server.
    Collects telemetry from diverse edge vehicles and builds a dynamic City-Wide Risk Map.
    """
    def __init__(self):
        # Maps zone_id -> aggregated risk data
        self.global_risk_map = defaultdict(lambda: {'total_aggression': 0.0, 'count': 0, 'avg_risk': 0.0})
        
    def report_risk(self, zone_id, aggression_score, collision_risk):
        """Receives real-time telemetry from an edge client."""
        zone_data = self.global_risk_map[zone_id]
        zone_data['total_aggression'] += aggression_score
        zone_data['count'] += 1
        
        # Exponential moving average for recent risk vs historical risk could be used,
        # but for simulation we just keep a running average and spike for collisions.
        zone_data['avg_risk'] = (zone_data['avg_risk'] * 0.9) + (aggression_score * 0.1)
        
        # If there's an actual collision/critical event, the zone becomes a hotspot
        if collision_risk:
            zone_data['avg_risk'] += 1.0
            
    def get_zone_config(self, zone_id, base_config):
        """
        Federated Parameter Re-weighting:
        If a vehicle enters a known "High Risk" zone, the server tells the vehicle
        to lower its aggression tolerance (become more cautious).
        """
        zone_risk = self.global_risk_map[zone_id]['avg_risk']
        updated_config = {'behavioral_engine': dict(base_config['behavioral_engine'])}
        
        # Dynamic threshold tuning
        if zone_risk > 1.0:
            # Extreme Risk (Accident Hotspot): Highly sensitive, alerts over minor aggression
            updated_config['behavioral_engine']['aggression_threshold'] = max(0.3, base_config['behavioral_engine']['aggression_threshold'] - 0.4)
        elif zone_risk > 0.5:
            # Moderate Risk: Slightly more cautious than default
            updated_config['behavioral_engine']['aggression_threshold'] = max(0.4, base_config['behavioral_engine']['aggression_threshold'] - 0.2)
        else:
            # Normal: Use baseline config
            pass
            
        return updated_config, zone_risk

class FederatedClient:
    """
    Simulates the Edge Vehicle node in the federated network.
    """
    def __init__(self, client_id, server, base_config):
        self.client_id = client_id
        self.server = server
        self.base_config = base_config
        self.current_zone = "ZONE_A (Suburbs)" 
        self.frame_counter = 0
        
    def send_telemetry(self, aggression_score, collision_risk):
        self.server.report_risk(self.current_zone, aggression_score, collision_risk)
        
    def sync_config(self):
        updated_params, zone_risk = self.server.get_zone_config(self.current_zone, self.base_config)
        return updated_params, zone_risk
        
    def update_zone(self, frame_count):
        """Simulate the vehicle driving through different geographical zones over time"""
        self.frame_counter = frame_count
        if frame_count < 150:
            self.current_zone = "ZONE_A (Suburbs)"
        elif frame_count < 350:
            self.current_zone = "ZONE_B (Downtown Hotspot)"
        else:
            self.current_zone = "ZONE_C (Highway)"
