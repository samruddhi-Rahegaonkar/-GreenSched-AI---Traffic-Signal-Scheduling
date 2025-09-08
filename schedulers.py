from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import LinearRegression
import numpy as np
import heapq

class BaseScheduler:
    """Base class for all scheduling algorithms"""
    
    def __init__(self, name):
        self.name = name
        self.decisions_made = 0
        self.performance_history = []
    
    def schedule(self, lanes, current_time, context=None):
        """Override this method in subclasses"""
        raise NotImplementedError
    
    def record_performance(self, decision, outcome):
        """Record scheduling decision and outcome"""
        self.performance_history.append({
            'decision': decision,
            'outcome': outcome,
            'timestamp': self.decisions_made
        })
        self.decisions_made += 1

class FCFSScheduler(BaseScheduler):
    """First Come First Serve Scheduler"""
    
    def __init__(self):
        super().__init__("FCFS")
    
    def schedule(self, lanes, current_time, context=None):
        oldest_time = float('inf')
        selected_lane = None
        
        for lane_name, lane in lanes.items():
            if lane.vehicles and lane.vehicles[0].arrival_time < oldest_time:
                oldest_time = lane.vehicles[0].arrival_time
                selected_lane = lane_name
                
        return selected_lane

class SJFScheduler(BaseScheduler):
    """Shortest Job First Scheduler"""
    
    def __init__(self):
        super().__init__("SJF")
    
    def schedule(self, lanes, current_time, context=None):
        min_vehicles = float('inf')
        selected_lane = None
        
        for lane_name, lane in lanes.items():
            if 0 < len(lane.vehicles) < min_vehicles:
                min_vehicles = len(lane.vehicles)
                selected_lane = lane_name
                
        return selected_lane

class PriorityScheduler(BaseScheduler):
    """Priority-based Scheduler with Emergency Handling"""
    
    def __init__(self):
        super().__init__("Priority")
    
    def schedule(self, lanes, current_time, context=None):
        max_priority = -1
        selected_lane = None
        
        for lane_name, lane in lanes.items():
            if lane.vehicles:
                priority, _ = lane.get_priority_score(current_time)
                if priority > max_priority:
                    max_priority = priority
                    selected_lane = lane_name
                    
        return selected_lane

class RoundRobinScheduler(BaseScheduler):
    """Round Robin Scheduler"""
    
    def __init__(self):
        super().__init__("Round Robin")
        self.last_lane = None
        self.lane_order = ['NORTH', 'EAST', 'SOUTH', 'WEST']
    
    def schedule(self, lanes, current_time, context=None):
        if self.last_lane is None:
            self.last_lane = self.lane_order[0]
            return self.lane_order[0]
            
        try:
            current_idx = self.lane_order.index(self.last_lane)
            next_idx = (current_idx + 1) % len(self.lane_order)
            
            # Find next lane with vehicles
            for i in range(len(self.lane_order)):
                candidate_idx = (next_idx + i) % len(self.lane_order)
                candidate_lane = self.lane_order[candidate_idx]
                if lanes[candidate_lane].vehicles:
                    self.last_lane = candidate_lane
                    return candidate_lane
        except ValueError:
            pass
            
        return None

class MLScheduler(BaseScheduler):
    """Machine Learning based Scheduler"""
    
    def __init__(self):
        super().__init__("ML-Based")
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.training_data = []
        self.is_trained = False
        self.min_training_samples = 20
        
    def collect_data(self, lane_counts, selected_lane, wait_times, outcome_score):
        """Collect training data"""
        features = list(lane_counts.values()) + [sum(wait_times)]
        self.training_data.append({
            'features': features,
            'selected_lane': selected_lane,
            'outcome_score': outcome_score
        })
        
    def train_model(self):
        """Train the ML model"""
        if len(self.training_data) < self.min_training_samples:
            return False
            
        X = [data['features'] for data in self.training_data]
        y = [data['outcome_score'] for data in self.training_data]
        
        self.model.fit(X, y)
        self.is_trained = True
        return True
        
    def schedule(self, lanes, current_time, context=None):
        """ML-based lane selection"""
        lane_counts = {name: len(lane.vehicles) for name, lane in lanes.items()}
        
        if not self.is_trained:
            # Fallback to priority scheduling
            fallback = PriorityScheduler()
            return fallback.schedule(lanes, current_time)
            
        best_lane = None
        best_score = float('inf')
        
        for lane_name in lanes.keys():
            if lanes[lane_name].vehicles:
                # Simulate giving green to this lane
                temp_counts = lane_counts.copy()
                temp_counts[lane_name] = max(0, temp_counts[lane_name] - 3)
                
                wait_times = []
                for lane in lanes.values():
                    for vehicle in lane.vehicles:
                        wait_times.append(current_time - vehicle.arrival_time)
                
                features = list(temp_counts.values()) + [sum(wait_times)]
                predicted_score = self.model.predict([features])[0]
                
                if predicted_score < best_score:
                    best_score = predicted_score
                    best_lane = lane_name
                    
        return best_lane
