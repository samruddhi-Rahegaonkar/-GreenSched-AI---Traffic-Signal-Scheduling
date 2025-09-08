import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
from collections import deque, defaultdict
import random
import math
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import heapq
from enum import Enum

# Enums for better type safety
class VehicleType(Enum):
    CAR = "car"
    TRUCK = "truck" 
    EMERGENCY = "emergency"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"

class SignalState(Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

class Direction(Enum):
    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"

@dataclass
class Vehicle:
    """Enhanced vehicle class with realistic properties"""
    lane: str
    arrival_time: float
    vehicle_type: VehicleType = VehicleType.CAR
    id: str = field(default_factory=lambda: f"v_{random.randint(10000, 99999)}")
    priority: int = field(init=False)
    processing_time: float = field(init=False)
    wait_time: float = 0.0
    fuel_consumption: float = field(init=False)
    emissions: float = field(init=False)
    
    def __post_init__(self):
        self.priority = self._calculate_priority()
        self.processing_time = self._calculate_processing_time()
        self.fuel_consumption = self._calculate_fuel_consumption()
        self.emissions = self._calculate_emissions()
    
    def _calculate_priority(self) -> int:
        priorities = {
            VehicleType.EMERGENCY: 100,
            VehicleType.BUS: 20,
            VehicleType.TRUCK: 5,
            VehicleType.CAR: 1,
            VehicleType.MOTORCYCLE: 1
        }
        return priorities[self.vehicle_type]
    
    def _calculate_processing_time(self) -> float:
        """Time needed to clear intersection"""
        times = {
            VehicleType.EMERGENCY: 2.0,
            VehicleType.BUS: 4.0,
            VehicleType.TRUCK: 3.5,
            VehicleType.CAR: 2.5,
            VehicleType.MOTORCYCLE: 2.0
        }
        return times[self.vehicle_type] + random.uniform(-0.5, 0.5)
    
    def _calculate_fuel_consumption(self) -> float:
        """Fuel consumption per minute while waiting"""
        consumption = {
            VehicleType.EMERGENCY: 0.3,
            VehicleType.BUS: 0.8,
            VehicleType.TRUCK: 0.6,
            VehicleType.CAR: 0.2,
            VehicleType.MOTORCYCLE: 0.1
        }
        return consumption[self.vehicle_type]
    
    def _calculate_emissions(self) -> float:
        """CO2 emissions per minute while waiting (kg)"""
        emissions = {
            VehicleType.EMERGENCY: 0.02,
            VehicleType.BUS: 0.05,
            VehicleType.TRUCK: 0.04,
            VehicleType.CAR: 0.015,
            VehicleType.MOTORCYCLE: 0.008
        }
        return emissions[self.vehicle_type]
    
    @property
    def is_emergency(self) -> bool:
        return self.vehicle_type == VehicleType.EMERGENCY
    
    def get_visual_properties(self) -> Dict:
        """Get visual properties for rendering"""
        props = {
            VehicleType.CAR: {'color': '#3498db', 'size': 12, 'symbol': 'circle', 'emoji': '🚗'},
            VehicleType.TRUCK: {'color': '#e74c3c', 'size': 16, 'symbol': 'square', 'emoji': '🚛'},
            VehicleType.EMERGENCY: {'color': '#f39c12', 'size': 14, 'symbol': 'diamond', 'emoji': '🚨'},
            VehicleType.BUS: {'color': '#9b59b6', 'size': 18, 'symbol': 'square', 'emoji': '🚌'},
            VehicleType.MOTORCYCLE: {'color': '#1abc9c', 'size': 10, 'symbol': 'circle', 'emoji': '🏍️'}
        }
        return props[self.vehicle_type]

class TrafficLane:
    """Enhanced traffic lane with realistic behavior"""
    
    def __init__(self, name: str, direction: str, capacity: int = 50):
        self.name = name
        self.direction = direction
        self.capacity = capacity
        self.vehicles: deque[Vehicle] = deque()
        self.total_vehicles_processed = 0
        self.total_wait_time = 0.0
        self.total_fuel_consumed = 0.0
        self.total_emissions = 0.0
        self.current_signal = SignalState.RED
        self.last_green_time = 0
        self.congestion_history = deque(maxlen=10)
        self.average_arrival_rate = 0.0
        
    def add_vehicle(self, vehicle: Vehicle) -> bool:
        """Add vehicle if lane has capacity"""
        if len(self.vehicles) < self.capacity:
            self.vehicles.append(vehicle)
            return True
        return False
    
    def process_vehicles(self, green_duration: float, current_time: float) -> Tuple[int, List[Vehicle]]:
        """Process vehicles with realistic timing"""
        processed = 0
        processed_vehicles = []
        time_used = 0.0
        
        while self.vehicles and time_used < green_duration:
            vehicle = self.vehicles[0]
            
            # Check if we have enough time to process this vehicle
            if time_used + vehicle.processing_time <= green_duration:
                vehicle = self.vehicles.popleft()
                vehicle.wait_time = current_time - vehicle.arrival_time
                
                # Update statistics
                self.total_vehicles_processed += 1
                self.total_wait_time += vehicle.wait_time
                self.total_fuel_consumed += vehicle.fuel_consumption * (vehicle.wait_time / 60)
                self.total_emissions += vehicle.emissions * (vehicle.wait_time / 60)
                
                processed_vehicles.append(vehicle)
                processed += 1
                time_used += vehicle.processing_time
            else:
                break
        
        return processed, processed_vehicles
    
    def get_priority_score(self, current_time: float) -> Tuple[float, bool]:
        """Advanced priority calculation"""
        if not self.vehicles:
            return 0.0, False
        
        # Base factors
        vehicle_count = len(self.vehicles)
        congestion_factor = min(vehicle_count / self.capacity, 1.0)
        time_since_green = current_time - self.last_green_time
        
        # Emergency vehicles get absolute priority
        has_emergency = any(v.is_emergency for v in self.vehicles)
        if has_emergency:
            return 1000.0 + vehicle_count, True
        
        # Calculate weighted priority based on vehicle types
        priority_sum = sum(v.priority for v in self.vehicles)
        avg_wait_time = np.mean([current_time - v.arrival_time for v in self.vehicles])
        
        # Advanced scoring
        priority = (
            priority_sum * 0.3 +  # Vehicle type priorities
            vehicle_count * 2.0 +  # Queue length
            congestion_factor * 20.0 +  # Congestion penalty
            min(time_since_green * 0.2, 15.0) +  # Time since last green
            min(avg_wait_time * 0.1, 10.0)  # Average waiting time
        )
        
        return priority, False
    
    def get_congestion_level(self) -> str:
        """Get current congestion level"""
        ratio = len(self.vehicles) / self.capacity
        if ratio >= 0.8:
            return "SEVERE"
        elif ratio >= 0.6:
            return "HIGH"
        elif ratio >= 0.4:
            return "MODERATE"
        elif ratio >= 0.2:
            return "LOW"
        else:
            return "CLEAR"
    
    def update_arrival_rate(self, vehicles_added: int):
        """Update average arrival rate"""
        self.congestion_history.append(vehicles_added)
        if self.congestion_history:
            self.average_arrival_rate = np.mean(self.congestion_history)

class BaseScheduler:
    """Enhanced base scheduler with performance tracking"""
    
    def __init__(self, name: str):
        self.name = name
        self.decisions_made = 0
        self.performance_metrics = defaultdict(list)
        self.total_wait_time = 0.0
        self.total_vehicles_processed = 0
        self.emergency_response_times = []
        
    def schedule(self, lanes: Dict[str, TrafficLane], current_time: float, 
                context: Optional[Dict] = None) -> Optional[str]:
        """Override in subclasses"""
        raise NotImplementedError
    
    def record_performance(self, wait_time: float, vehicles_processed: int, 
                         emergency_handled: bool = False, emergency_wait: float = 0.0):
        """Record performance metrics"""
        self.total_wait_time += wait_time
        self.total_vehicles_processed += vehicles_processed
        self.decisions_made += 1
        
        if emergency_handled:
            self.emergency_response_times.append(emergency_wait)
        
        self.performance_metrics['avg_wait_time'].append(
            self.total_wait_time / max(self.total_vehicles_processed, 1)
        )
        self.performance_metrics['throughput'].append(
            self.total_vehicles_processed / max(self.decisions_made, 1)
        )
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        if not self.performance_metrics['avg_wait_time']:
            return {'avg_wait_time': 0, 'throughput': 0, 'emergency_response': 0}
        
        return {
            'avg_wait_time': np.mean(self.performance_metrics['avg_wait_time']),
            'throughput': np.mean(self.performance_metrics['throughput']),
            'emergency_response': np.mean(self.emergency_response_times) if self.emergency_response_times else 0,
            'decisions_made': self.decisions_made
        }

class AdaptivePriorityScheduler(BaseScheduler):
    """Adaptive priority scheduler with learning capabilities"""
    
    def __init__(self):
        super().__init__("Adaptive Priority")
        self.congestion_weights = defaultdict(lambda: 1.0)
        self.time_of_day_patterns = defaultdict(list)
        
    def schedule(self, lanes: Dict[str, TrafficLane], current_time: float, 
                context: Optional[Dict] = None) -> Optional[str]:
        max_priority = -1
        selected_lane = None
        
        for lane_name, lane in lanes.items():
            if lane.vehicles:
                priority, is_emergency = lane.get_priority_score(current_time)
                
                # Apply adaptive weights based on historical congestion
                adaptive_weight = self.congestion_weights[lane_name]
                adjusted_priority = priority * adaptive_weight
                
                if adjusted_priority > max_priority:
                    max_priority = adjusted_priority
                    selected_lane = lane_name
        
        # Update congestion weights based on current state
        self._update_weights(lanes)
        
        return selected_lane
    
    def _update_weights(self, lanes: Dict[str, TrafficLane]):
        """Update adaptive weights based on congestion levels"""
        total_vehicles = sum(len(lane.vehicles) for lane in lanes.values())
        
        for lane_name, lane in lanes.items():
            congestion_ratio = len(lane.vehicles) / max(total_vehicles, 1)
            
            # Increase weight for more congested lanes
            self.congestion_weights[lane_name] = 0.7 * self.congestion_weights[lane_name] + 0.3 * (1 + congestion_ratio)

class MLEnhancedScheduler(BaseScheduler):
    """Advanced ML scheduler with multiple algorithms"""
    
    def __init__(self):
        super().__init__("ML Enhanced")
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression()
        }
        self.scaler = StandardScaler()
        self.training_data = []
        self.is_trained = False
        self.feature_importance = None
        self.model_performance = defaultdict(list)
        self.best_model = 'rf'
        
    def _extract_features(self, lanes: Dict[str, TrafficLane], current_time: float) -> List[float]:
        """Extract comprehensive features"""
        features = []
        
        # Basic lane features
        for lane_name in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            lane = lanes[lane_name]
            features.extend([
                len(lane.vehicles),  # Queue length
                lane.get_priority_score(current_time)[0],  # Priority score
                sum(v.priority for v in lane.vehicles),  # Total priority
                np.mean([current_time - v.arrival_time for v in lane.vehicles]) if lane.vehicles else 0,  # Avg wait
                len([v for v in lane.vehicles if v.is_emergency]),  # Emergency count
                lane.average_arrival_rate,  # Arrival rate
                current_time - lane.last_green_time  # Time since green
            ])
        
        # Global features
        total_vehicles = sum(len(lane.vehicles) for lane in lanes.values())
        total_emergency = sum(len([v for v in lane.vehicles if v.is_emergency]) for lane in lanes.values())
        
        features.extend([
            total_vehicles,
            total_emergency,
            current_time % (24 * 60),  # Time of day (minutes)
            current_time % (7 * 24 * 60)  # Time of week (minutes)
        ])
        
        return features
    
    def collect_data(self, lanes: Dict[str, TrafficLane], selected_lane: str, 
                    current_time: float, outcome_metrics: Dict):
        """Collect training data with rich features"""
        features = self._extract_features(lanes, current_time)
        
        # Outcome score (lower is better)
        outcome_score = (
            outcome_metrics.get('avg_wait_time', 0) * 0.4 +
            outcome_metrics.get('total_fuel', 0) * 0.3 +
            outcome_metrics.get('total_emissions', 0) * 0.3
        )
        
        self.training_data.append({
            'features': features,
            'selected_lane': selected_lane,
            'outcome_score': outcome_score,
            'timestamp': current_time
        })
        
        # Keep only recent data to adapt to changing patterns
        if len(self.training_data) > 1000:
            self.training_data = self.training_data[-800:]
    
    def train_models(self) -> bool:
        """Train all models and select the best one"""
        if len(self.training_data) < 50:
            return False
        
        # Prepare data
        X = np.array([data['features'] for data in self.training_data])
        y = np.array([data['outcome_score'] for data in self.training_data])
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train and evaluate models
        best_score = float('inf')
        
        for model_name, model in self.models.items():
            try:
                model.fit(X_scaled, y)
                predictions = model.predict(X_scaled)
                mse = np.mean((predictions - y) ** 2)
                
                self.model_performance[model_name].append(mse)
                
                if mse < best_score:
                    best_score = mse
                    self.best_model = model_name
                    
            except Exception as e:
                continue
        
        # Extract feature importance from random forest
        if hasattr(self.models['rf'], 'feature_importances_'):
            self.feature_importance = self.models['rf'].feature_importances_
        
        self.is_trained = True
        return True
    
    def schedule(self, lanes: Dict[str, TrafficLane], current_time: float, 
                context: Optional[Dict] = None) -> Optional[str]:
        """ML-based scheduling with fallback"""
        if not self.is_trained:
            # Fallback to adaptive priority
            fallback = AdaptivePriorityScheduler()
            return fallback.schedule(lanes, current_time)
        
        # Extract features
        features = self._extract_features(lanes, current_time)
        features_scaled = self.scaler.transform([features])
        
        best_lane = None
        best_score = float('inf')
        
        # Evaluate each possible lane selection
        for lane_name in lanes.keys():
            if lanes[lane_name].vehicles:
                try:
                    # Use the best performing model
                    model = self.models[self.best_model]
                    predicted_score = model.predict(features_scaled)[0]
                    
                    # Add some randomness to prevent getting stuck in local optima
                    adjusted_score = predicted_score + random.uniform(-0.1, 0.1)
                    
                    if adjusted_score < best_score:
                        best_score = adjusted_score
                        best_lane = lane_name
                        
                except Exception:
                    continue
        
        return best_lane

class EnhancedTrafficSimulation:
    """Advanced traffic simulation with comprehensive analytics"""
    
    def __init__(self):
        self.lanes = {
            direction.value: TrafficLane(direction.value, direction.value) 
            for direction in Direction
        }
        self.current_time = 0.0
        self.schedulers = self._initialize_schedulers()
        self.current_scheduler = None
        self.simulation_history = []
        self.environmental_metrics = defaultdict(float)
        self.vehicle_spawn_patterns = self._initialize_spawn_patterns()
        
    def _initialize_schedulers(self) -> Dict[str, BaseScheduler]:
        """Initialize all available schedulers"""
        return {
            'Adaptive Priority': AdaptivePriorityScheduler(),
            'ML Enhanced': MLEnhancedScheduler(),
            'FCFS': self._create_fcfs_scheduler(),
            'Round Robin': self._create_round_robin_scheduler()
        }
    
    def _create_fcfs_scheduler(self) -> BaseScheduler:
        """Create FCFS scheduler"""
        class FCFSScheduler(BaseScheduler):
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
        
        return FCFSScheduler()
    
    def _create_round_robin_scheduler(self) -> BaseScheduler:
        """Create Round Robin scheduler"""
        class RoundRobinScheduler(BaseScheduler):
            def __init__(self):
                super().__init__("Round Robin")
                self.last_lane_index = -1
                self.lane_order = list(Direction)
                
            def schedule(self, lanes, current_time, context=None):
                for i in range(len(self.lane_order)):
                    self.last_lane_index = (self.last_lane_index + 1) % len(self.lane_order)
                    candidate_lane = self.lane_order[self.last_lane_index].value
                    if lanes[candidate_lane].vehicles:
                        return candidate_lane
                return None
        
        return RoundRobinScheduler()
    
    def _initialize_spawn_patterns(self) -> Dict:
        """Initialize realistic vehicle spawn patterns"""
        return {
            'base_rate': 0.3,
            'peak_hours': [7, 8, 9, 17, 18, 19],  # Rush hours
            'vehicle_distribution': {
                VehicleType.CAR: 0.70,
                VehicleType.TRUCK: 0.15,
                VehicleType.BUS: 0.10,
                VehicleType.MOTORCYCLE: 0.04,
                VehicleType.EMERGENCY: 0.01
            }
        }
    
    def generate_realistic_vehicles(self):
        """Generate vehicles with realistic patterns"""
        # Time-based spawn rate adjustment
        hour = int((self.current_time / 60) % 24)
        base_rate = self.vehicle_spawn_patterns['base_rate']
        
        if hour in self.vehicle_spawn_patterns['peak_hours']:
            spawn_rate = base_rate * 1.8
        elif 22 <= hour or hour <= 5:  # Late night
            spawn_rate = base_rate * 0.3
        else:
            spawn_rate = base_rate
        
        for lane_name, lane in self.lanes.items():
            if random.random() < spawn_rate:
                # Select vehicle type based on distribution
                rand = random.random()
                cumulative = 0
                vehicle_type = VehicleType.CAR  # Default
                
                for vtype, prob in self.vehicle_spawn_patterns['vehicle_distribution'].items():
                    cumulative += prob
                    if rand <= cumulative:
                        vehicle_type = vtype
                        break
                
                vehicle = Vehicle(lane_name, self.current_time, vehicle_type)
                lane.add_vehicle(vehicle)
                
                # Update arrival rate tracking
                lane.update_arrival_rate(1)
    
    def simulate_step(self, green_duration: float = 30.0) -> Optional[Dict]:
        """Enhanced simulation step with comprehensive metrics"""
        if not self.current_scheduler:
            return None
        
        self.current_time += 1.0
        
        # Generate vehicles
        self.generate_realistic_vehicles()
        
        # Get scheduling decision
        selected_lane = self.current_scheduler.schedule(self.lanes, self.current_time)
        
        # Process vehicles
        vehicles_processed = 0
        processed_vehicles = []
        total_wait_time = 0.0
        emergency_handled = False
        emergency_wait_time = 0.0
        
        if selected_lane and self.lanes[selected_lane].vehicles:
            vehicles_processed, processed_vehicles = self.lanes[selected_lane].process_vehicles(
                green_duration, self.current_time
            )
            
            # Update signals
            self.lanes[selected_lane].current_signal = SignalState.GREEN
            self.lanes[selected_lane].last_green_time = self.current_time
            
            for lane_name, lane in self.lanes.items():
                if lane_name != selected_lane:
                    lane.current_signal = SignalState.RED
            
            # Calculate metrics for processed vehicles
            for vehicle in processed_vehicles:
                total_wait_time += vehicle.wait_time
                if vehicle.is_emergency:
                    emergency_handled = True
                    emergency_wait_time = vehicle.wait_time
        
        # Calculate comprehensive metrics
        lane_counts = {name: len(lane.vehicles) for name, lane in self.lanes.items()}
        
        # Current waiting vehicles metrics
        all_wait_times = []
        total_fuel_consumption = 0.0
        total_emissions = 0.0
        
        for lane in self.lanes.values():
            for vehicle in lane.vehicles:
                wait_time = self.current_time - vehicle.arrival_time
                all_wait_times.append(wait_time)
                total_fuel_consumption += vehicle.fuel_consumption * (wait_time / 60)
                total_emissions += vehicle.emissions * (wait_time / 60)
        
        avg_wait_time = np.mean(all_wait_times) if all_wait_times else 0
        
        # Update environmental metrics
        self.environmental_metrics['total_fuel'] += total_fuel_consumption
        self.environmental_metrics['total_emissions'] += total_emissions
        
        # Record performance
        outcome_metrics = {
            'avg_wait_time': avg_wait_time,
            'total_fuel': total_fuel_consumption,
            'total_emissions': total_emissions
        }
        
        self.current_scheduler.record_performance(
            total_wait_time, vehicles_processed, emergency_handled, emergency_wait_time
        )
        
        # Collect ML training data
        if isinstance(self.current_scheduler, MLEnhancedScheduler) and selected_lane:
            self.current_scheduler.collect_data(
                self.lanes, selected_lane, self.current_time, outcome_metrics
            )
        
        # Create comprehensive step data
        step_data = {
            'time': self.current_time,
            'scheduler': self.current_scheduler.name,
            'selected_lane': selected_lane,
            'vehicles_processed': vehicles_processed,
            'lane_counts': lane_counts.copy(),
            'avg_wait_time': avg_wait_time,
            'total_vehicles': sum(lane_counts.values()),
            'processed_vehicles': processed_vehicles,
            'fuel_consumption': total_fuel_consumption,
            'emissions': total_emissions,
            'congestion_levels': {name: lane.get_congestion_level() for name, lane in self.lanes.items()},
            'emergency_handled': emergency_handled
        }
        
        self.simulation_history.append(step_data)
        
        # Train ML model periodically
        if (isinstance(self.current_scheduler, MLEnhancedScheduler) and 
            len(self.current_scheduler.training_data) % 100 == 0):
            self.current_scheduler.train_models()
        
        return step_data
    
    def set_scheduler(self, scheduler_name: str) -> bool:
        """Set the current scheduler"""
        if scheduler_name in self.schedulers:
            self.current_scheduler = self.schedulers[scheduler_name]
            return True
        return False
    
    def get_comprehensive_analytics(self) -> Dict:
        """Get comprehensive performance analytics"""
        if not self.simulation_history:
            return {}
        
        df = pd.DataFrame(self.simulation_history)
        
        analytics = {
            'performance': {
                'avg_wait_time': df['avg_wait_time'].mean(),
                'total_throughput': df['vehicles_processed'].sum(),
                'avg_throughput': df['vehicles_processed'].mean(),
                'emergency_response_rate': (df['emergency_handled'].sum() / max(len(df), 1)) * 100
            },
            'environmental': {
                'total_fuel_consumption': self.environmental_metrics['total_fuel'],
                'total_emissions': self.environmental_metrics['total_emissions'],
                'avg_fuel_per_vehicle': self.environmental_metrics['total_fuel'] / max(df['vehicles_processed'].sum(), 1),
                'avg_emissions_per_vehicle': self.environmental_metrics['total_emissions'] / max(df['vehicles_processed'].sum(), 1)
            },
            'efficiency': {
                'congestion_incidents': sum(1 for step in self.simulation_history 
                                          if any(level in ['HIGH', 'SEVERE'] 
                                               for level in step['congestion_levels'].values())),
                'peak_queue_length': max(step['total_vehicles'] for step in self.simulation_history),
                'scheduler_performance': self.current_scheduler.get_performance_summary() if self.current_scheduler else {}
            }
        }
        
        return analytics
    
    def reset(self):
        """Reset simulation with enhanced cleanup"""
        self.current_time = 0.0
        self.simulation_history.clear()
        self.environmental_metrics.clear()
        
        # Reset all lanes
        for lane in self.lanes.values():
            lane.vehicles.clear()
            lane.total_vehicles_processed = 0
            lane.total_wait_time = 0.0
            lane.total_fuel_consumed = 0.0
            lane.total_emissions = 0.0
            lane.current_signal = SignalState.RED
            lane.last_green_time = 0
            lane.congestion_history.clear()
            lane.average_arrival_rate = 0.0
        
        # Reset schedulers
        self.schedulers = self._initialize_schedulers()

# Streamlit UI with enhanced visualization
def create_enhanced_intersection_plot(simulation: EnhancedTrafficSimulation):
    """Create enhanced intersection visualization"""
    fig = go.Figure()
    
    # Enhanced road design
    fig.add_shape(type="rect", x0=-2, y0=-0.2, x1=2, y1=0.2, 
                  fillcolor="#2c3e50", line_color="#34495e", line_width=2)
    fig.add_shape(type="rect", x0=-0.2, y0=-2, x1=0.2, y1=2, 
                  fillcolor="#2c3e50", line_color="#34495e", line_width=2)
    
    # Lane markings and intersection box
    fig.add_shape(type="rect", x0=-0.2, y0=-0.2, x1=0.2, y1=0.2, 
                  fillcolor="#34495e", line_color="white", line_width=1)
    
    # Traffic lights with enhanced positioning
    positions = {
        'NORTH': (0.4, 0.4, 'bottom center'),
        'SOUTH': (-0.4, -0.4, 'top center'),
        'EAST': (0.4, -0.4, 'middle left'),
        'WEST': (-0.4, 0.4, 'middle right')
    }
    
    # Vehicle queue positions
    queue_positions = {
        'NORTH': {'start_x': 0, 'start_y': 0.3, 'dx': 0.15, 'dy': 0.2},
        'SOUTH': {'start_x': 0, 'start_y': -0.3, 'dx': -0.15, 'dy': -0.2},
        'EAST': {'start_x': 0.3, 'start_y': 0, 'dx': 0.2, 'dy': 0.15},
        'WEST': {'start_x': -0.3, 'start_y': 0, 'dx': -0.2, 'dy': -0.15}
    }
    
    # Draw traffic lights and lane information
    for lane_name, (x, y, textpos) in positions.items():
        lane = simulation.lanes[lane_name]
        signal_color = {
            SignalState.GREEN: '#27ae60',
            SignalState.YELLOW: '#f1c40f', 
            SignalState.RED: '#e74c3c'
        }[lane.current_signal]
        
        # Traffic light
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=25, color=signal_color, 
                       line=dict(width=3, color='black'),
                       symbol='square'),
            name=f'{lane_name}_light',
            showlegend=False
        ))
        
        # Lane info with congestion level
        congestion_level = lane.get_congestion_level()
        congestion_colors = {
            'CLEAR': '🟢', 'LOW': '🟡', 'MODERATE': '🟠', 'HIGH': '🔴', 'SEVERE': '⚫'
        }
        
        vehicle_count = len(lane.vehicles)
        emergency_count = sum(1 for v in lane.vehicles if v.is_emergency)
        
        info_text = f"{lane_name}<br>{vehicle_count} vehicles<br>{congestion_colors[congestion_level]} {congestion_level}"
        if emergency_count > 0:
            info_text += f"<br>🚨 {emergency_count} emergency"
        
        fig.add_annotation(
            x=x*1.8, y=y*1.8, text=info_text,
            showarrow=False, font=dict(size=10, color='white'),
            bgcolor='rgba(0,0,0,0.8)', bordercolor='white', borderwidth=1
        )
    
    # Draw individual vehicles
    all_vehicles_data = {'x': [], 'y': [], 'colors': [], 'sizes': [], 'symbols': [], 'text': []}
    
    for lane_name, lane in simulation.lanes.items():
        pos_config = queue_positions[lane_name]
        
        for i, vehicle in enumerate(list(lane.vehicles)[:10]):  # Show max 10 vehicles per lane
            props = vehicle.get_visual_properties()
            
            # Calculate position
            x = pos_config['start_x'] + (i * pos_config['dx'])
            y = pos_config['start_y'] + (i * pos_config['dy'])
            
            all_vehicles_data['x'].append(x)
            all_vehicles_data['y'].append(y)
            all_vehicles_data['colors'].append(props['color'])
            all_vehicles_data['sizes'].append(props['size'])
            all_vehicles_data['symbols'].append(props['symbol'])
            all_vehicles_data['text'].append(props['emoji'])
    
    # Add vehicles to plot
    if all_vehicles_data['x']:
        fig.add_trace(go.Scatter(
            x=all_vehicles_data['x'], y=all_vehicles_data['y'],
            mode='markers+text',
            marker=dict(
                size=all_vehicles_data['sizes'],
                color=all_vehicles_data['colors'],
                line=dict(width=2, color='white'),
                symbol=all_vehicles_data['symbols']
            ),
            text=all_vehicles_data['text'],
            textposition='middle center',
            textfont=dict(size=8),
            name='Vehicles',
            showlegend=False
        ))
    
    # Add directional arrows
    arrow_positions = [
        (0, 1.7, 0, -0.3, 'NORTH ↓'),
        (0, -1.7, 0, 0.3, 'SOUTH ↑'),
        (1.7, 0, -0.3, 0, 'EAST ←'),
        (-1.7, 0, 0.3, 0, 'WEST →')
    ]
    
    for x, y, dx, dy, label in arrow_positions:
        fig.add_annotation(
            x=x, y=y, ax=x+dx*100, ay=y+dy*100,
            arrowhead=2, arrowsize=2, arrowwidth=3,
            arrowcolor='white', text=label,
            font=dict(size=12, color='white'),
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='white', borderwidth=1
        )
    
    fig.update_layout(
        xaxis=dict(range=[-2.5, 2.5], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-2.5, 2.5], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='#1a252f',
        paper_bgcolor='#1a252f',
        height=600,
        title=dict(
            text="🚦 Enhanced Traffic Intersection - Real-time Simulation",
            font=dict(color='white', size=18),
            x=0.5
        )
    )
    
    return fig

def create_performance_dashboard(analytics: Dict):
    """Create comprehensive performance dashboard"""
    if not analytics:
        return go.Figure()
    
    fig = go.Figure()
    
    # Performance metrics as gauge charts
    performance = analytics.get('performance', {})
    environmental = analytics.get('environmental', {})
    
    # Add gauge for average wait time
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=performance.get('avg_wait_time', 0),
        domain={'x': [0, 0.33], 'y': [0.5, 1]},
        title={'text': "Avg Wait Time (s)"},
        gauge={
            'axis': {'range': [None, 120]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 60], 'color': "gray"},
                {'range': [60, 120], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Add gauge for throughput
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=performance.get('avg_throughput', 0),
        domain={'x': [0.33, 0.66], 'y': [0.5, 1]},
        title={'text': "Throughput (veh/cycle)"},
        gauge={
            'axis': {'range': [None, 10]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 3], 'color': "lightgray"},
                {'range': [3, 6], 'color': "gray"},
                {'range': [6, 10], 'color': "darkgray"}
            ]
        }
    ))
    
    # Add gauge for emissions
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=environmental.get('total_emissions', 0),
        domain={'x': [0.66, 1], 'y': [0.5, 1]},
        title={'text': "Total Emissions (kg CO2)"},
        gauge={
            'axis': {'range': [None, 50]},
            'bar': {'color': "orange"},
            'steps': [
                {'range': [0, 15], 'color': "lightgray"},
                {'range': [15, 30], 'color': "gray"},
                {'range': [30, 50], 'color': "darkgray"}
            ]
        }
    ))
    
    # Add text metrics
    metrics_text = f"""
    Emergency Response Rate: {performance.get('emergency_response_rate', 0):.1f}%
    Total Fuel Consumption: {environmental.get('total_fuel_consumption', 0):.2f} L
    Peak Queue Length: {analytics.get('efficiency', {}).get('peak_queue_length', 0)} vehicles
    Congestion Incidents: {analytics.get('efficiency', {}).get('congestion_incidents', 0)}
    """
    
    fig.add_annotation(
        x=0.5, y=0.4, text=metrics_text,
        showarrow=False, font=dict(size=14, color='white'),
        bgcolor='rgba(0,0,0,0.8)', bordercolor='white', borderwidth=1,
        align='left'
    )
    
    fig.update_layout(
        plot_bgcolor='#1a252f',
        paper_bgcolor='#1a252f',
        height=400,
        title=dict(
            text="📊 Performance Dashboard",
            font=dict(color='white', size=16),
            x=0.5
        ),
        font=dict(color='white')
    )
    
    return fig

# Main Streamlit Application
st.set_page_config(
    page_title="Enhanced GreenSched AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'simulation' not in st.session_state:
    st.session_state.simulation = EnhancedTrafficSimulation()
    st.session_state.running = False
    st.session_state.step_count = 0

def main():
    simulation = st.session_state.simulation
    
    # Header
    st.title("🚦 Enhanced GreenSched AI")
    st.markdown("### Advanced Traffic Signal Optimization with ML & Environmental Analytics")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Simulation Controls")
        
        # Scheduler selection
        scheduler_options = list(simulation.schedulers.keys())
        selected_scheduler = st.selectbox(
            "Select Scheduling Algorithm",
            scheduler_options,
            index=0
        )
        
        if simulation.current_scheduler is None or simulation.current_scheduler.name != selected_scheduler:
            simulation.set_scheduler(selected_scheduler)
        
        # Simulation parameters
        st.subheader("⚙️ Parameters")
        green_duration = st.slider("Green Light Duration (seconds)", 15, 120, 30)
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            st.write("Vehicle Spawn Patterns:")
            spawn_multiplier = st.slider("Traffic Intensity", 0.5, 2.0, 1.0)
            simulation.vehicle_spawn_patterns['base_rate'] = 0.3 * spawn_multiplier
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.running = True
        with col2:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.running = False
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                simulation.reset()
                st.session_state.running = False
                st.session_state.step_count = 0
        
        # Simulation status
        st.subheader("📊 Status")
        st.metric("Simulation Time", f"{simulation.current_time:.0f}s")
        st.metric("Steps Completed", st.session_state.step_count)
        st.metric("Current Scheduler", selected_scheduler)
        
        if simulation.current_scheduler:
            perf = simulation.current_scheduler.get_performance_summary()
            st.metric("Decisions Made", perf.get('decisions_made', 0))
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚦 Live Intersection View")
        intersection_fig = create_enhanced_intersection_plot(simulation)
        st.plotly_chart(intersection_fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Real-time Metrics")
        
        if simulation.simulation_history:
            latest = simulation.simulation_history[-1]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Active Lane", latest.get('selected_lane', 'None'))
                st.metric("Total Vehicles", latest.get('total_vehicles', 0))
                st.metric("Emergency Handled", "✅" if latest.get('emergency_handled', False) else "❌")
            
            with col_b:
                st.metric("Avg Wait Time", f"{latest.get('avg_wait_time', 0):.1f}s")
                st.metric("Processed", latest.get('vehicles_processed', 0))
                st.metric("Fuel Used", f"{latest.get('fuel_consumption', 0):.2f}L")
        
        # Queue status with enhanced information
        st.subheader("🚦 Live Queue Status")
        for lane_name, lane in simulation.lanes.items():
            with st.container():
                signal_emoji = "🟢" if lane.current_signal == SignalState.GREEN else "🔴"
                congestion = lane.get_congestion_level()
                
                # Count vehicle types
                vehicle_counts = defaultdict(int)
                for vehicle in lane.vehicles:
                    vehicle_counts[vehicle.vehicle_type] += 1
                
                # Format vehicle breakdown
                breakdown = []
                for vtype, count in vehicle_counts.items():
                    if count > 0:
                        emoji = Vehicle("", 0, vtype).get_visual_properties()['emoji']
                        breakdown.append(f"{emoji}{count}")
                
                breakdown_text = " ".join(breakdown) if breakdown else "Empty"
                
                st.write(f"{signal_emoji} **{lane_name}** ({congestion.lower()}): {breakdown_text}")
    
    # Performance Dashboard
    analytics = simulation.get_comprehensive_analytics()
    if analytics:
        st.subheader("📊 Performance Dashboard")
        dashboard_fig = create_performance_dashboard(analytics)
        st.plotly_chart(dashboard_fig, use_container_width=True)
        
        # Detailed analytics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⚡ Performance")
            perf = analytics.get('performance', {})
            st.metric("Average Wait Time", f"{perf.get('avg_wait_time', 0):.1f}s")
            st.metric("Total Throughput", f"{perf.get('total_throughput', 0):.0f}")
            st.metric("Emergency Response", f"{perf.get('emergency_response_rate', 0):.1f}%")
        
        with col2:
            st.subheader("🌍 Environmental")
            env = analytics.get('environmental', {})
            st.metric("Total Fuel", f"{env.get('total_fuel_consumption', 0):.2f}L")
            st.metric("Total CO2", f"{env.get('total_emissions', 0):.2f}kg")
            st.metric("Avg Fuel/Vehicle", f"{env.get('avg_fuel_per_vehicle', 0):.3f}L")
        
        with col3:
            st.subheader("🎯 Efficiency")
            eff = analytics.get('efficiency', {})
            st.metric("Congestion Events", eff.get('congestion_incidents', 0))
            st.metric("Peak Queue", f"{eff.get('peak_queue_length', 0)} vehicles")
    
    # Historical performance charts
    if len(simulation.simulation_history) > 10:
        st.subheader("📈 Historical Performance")
        
        df = pd.DataFrame(simulation.simulation_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.line(df.tail(50), x='time', y='avg_wait_time',
                          title='Average Wait Time Trend',
                          color_discrete_sequence=['#3498db'])
            fig1.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.line(df.tail(50), x='time', y='vehicles_processed',
                          title='Throughput Over Time',
                          color_discrete_sequence=['#27ae60'])
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Environmental impact over time
        if 'emissions' in df.columns:
            fig3 = px.area(df.tail(50), x='time', y='emissions',
                          title='Cumulative Environmental Impact',
                          color_discrete_sequence=['#e74c3c'])
            fig3.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig3, use_container_width=True)
    
    # ML Model insights (if using ML scheduler)
    if (isinstance(simulation.current_scheduler, MLEnhancedScheduler) and 
        simulation.current_scheduler.is_trained):
        
        with st.expander("🧠 ML Model Insights"):
            ml_scheduler = simulation.current_scheduler
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Performance:**")
                for model_name, scores in ml_scheduler.model_performance.items():
                    if scores:
                        st.metric(f"{model_name.upper()} MSE", f"{scores[-1]:.4f}")
                
                st.write(f"**Active Model:** {ml_scheduler.best_model.upper()}")
                st.write(f"**Training Samples:** {len(ml_scheduler.training_data)}")
            
            with col2:
                if ml_scheduler.feature_importance is not None:
                    st.write("**Top Feature Importances:**")
                    feature_names = [
                        "North Queue", "North Priority", "North Total Priority", "North Avg Wait",
                        "North Emergency", "North Arrival Rate", "North Time Since Green",
                        "South Queue", "South Priority", "South Total Priority", "South Avg Wait",
                        "South Emergency", "South Arrival Rate", "South Time Since Green",
                        "East Queue", "East Priority", "East Total Priority", "East Avg Wait", 
                        "East Emergency", "East Arrival Rate", "East Time Since Green",
                        "West Queue", "West Priority", "West Total Priority", "West Avg Wait",
                        "West Emergency", "West Arrival Rate", "West Time Since Green",
                        "Total Vehicles", "Total Emergency", "Time of Day", "Time of Week"
                    ]
                    
                    importance_df = pd.DataFrame({
                        'Feature': feature_names[:len(ml_scheduler.feature_importance)],
                        'Importance': ml_scheduler.feature_importance
                    }).sort_values('Importance', ascending=False).head(8)
                    
                    fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                                          orientation='h', title='Feature Importance')
                    fig_importance.update_layout(height=300)
                    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Vehicle type legend
    with st.expander("🚗 Vehicle Types & Legend"):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        vehicle_types = [
            (VehicleType.CAR, "Regular Cars", "70% of traffic"),
            (VehicleType.TRUCK, "Trucks", "15% of traffic"),
            (VehicleType.BUS, "Buses", "10% of traffic"),
            (VehicleType.MOTORCYCLE, "Motorcycles", "4% of traffic"),
            (VehicleType.EMERGENCY, "Emergency", "1% of traffic")
        ]
        
        for col, (vtype, name, description) in zip([col1, col2, col3, col4, col5], vehicle_types):
            with col:
                props = Vehicle("", 0, vtype).get_visual_properties()
                st.markdown(f"""
                **{props['emoji']} {name}**  
                {description}  
                Priority: {Vehicle('', 0, vtype).priority}
                """)
    
    # Auto-run simulation
    if st.session_state.running:
        time.sleep(0.5)  # Adjust speed for better visualization
        step_result = simulation.simulate_step(green_duration)
        if step_result:
            st.session_state.step_count += 1
        st.rerun()

if __name__ == "__main__":
    main()