import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
from collections import deque
import random
import math

# Enhanced Vehicle class with visual properties
class Vehicle:
    def __init__(self, lane, arrival_time, vehicle_type='car'):
        self.lane = lane
        self.arrival_time = arrival_time
        self.vehicle_type = vehicle_type
        self.wait_time = 0
        self.is_emergency = vehicle_type == 'emergency'
        self.id = f"{lane}_{arrival_time}_{random.randint(1000, 9999)}"
        self.position = self.get_initial_position()
        self.color = self.get_vehicle_color()
        self.size = self.get_vehicle_size()
        self.emoji = self.get_vehicle_emoji()
        
    def get_initial_position(self):
        """Get initial position based on lane - positioned at lane entrance"""
        positions = {
            'NORTH': {'x': 0, 'y': 0.9},
            'SOUTH': {'x': 0, 'y': -0.9},
            'EAST': {'x': 0.9, 'y': 0},
            'WEST': {'x': -0.9, 'y': 0}
        }
        return positions.get(self.lane, {'x': 0, 'y': 0})
    
    def get_vehicle_color(self):
        """Get vehicle color based on type"""
        colors = {
            'car': '#3498db',      # Blue
            'truck': '#e74c3c',    # Red
            'emergency': '#f39c12'  # Orange
        }
        return colors.get(self.vehicle_type, '#95a5a6')
    
    def get_vehicle_size(self):
        """Get vehicle size based on type"""
        sizes = {
            'car': 12,
            'truck': 16,
            'emergency': 14
        }
        return sizes.get(self.vehicle_type, 12)

    def get_vehicle_emoji(self):
        """Get vehicle emoji based on type"""
        emojis = {
            'car': '🚗',
            'truck': '🚛',
            'emergency': '🚨'
        }
        return emojis.get(self.vehicle_type, '🚗')

class VisualTrafficLane:
    def __init__(self, name, direction):
        self.name = name
        self.direction = direction
        self.vehicles = deque()
        self.total_vehicles_processed = 0
        self.total_wait_time = 0
        self.current_signal = 'RED'
        self.last_green_time = 0
        self.queue_positions = self.get_queue_positions()
        
    def get_queue_positions(self):
        """Define where vehicles queue for each lane - positioned clearly on lanes"""
        positions = {
            'NORTH': {'start_x': 0, 'start_y': 0.4, 'dx': 0, 'dy': 0.12},
            'SOUTH': {'start_x': 0, 'start_y': -0.4, 'dx': 0, 'dy': -0.12},
            'EAST': {'start_x': 0.4, 'start_y': 0, 'dx': 0.12, 'dy': 0},
            'WEST': {'start_x': -0.4, 'start_y': 0, 'dx': -0.12, 'dy': 0}
        }
        return positions.get(self.name, {'start_x': 0, 'start_y': 0, 'dx': 0, 'dy': 0})
        
    def add_vehicle(self, vehicle):
        self.vehicles.append(vehicle)
        self.update_vehicle_positions()
    
    def update_vehicle_positions(self):
        """Update positions of vehicles in queue"""
        for i, vehicle in enumerate(self.vehicles):
            pos = self.queue_positions
            vehicle.position = {
                'x': pos['start_x'] + i * pos['dx'],
                'y': pos['start_y'] + i * pos['dy']
            }
    
    def process_vehicles(self, green_duration, current_time):
        processed = 0
        vehicles_per_second = 1.0  # Increased processing rate
        max_vehicles = int(green_duration * vehicles_per_second)

        processed_vehicles = []

        while self.vehicles and processed < max_vehicles:
            vehicle = self.vehicles.popleft()
            vehicle.wait_time = current_time - vehicle.arrival_time
            self.total_vehicles_processed += 1
            self.total_wait_time += vehicle.wait_time

            # Update position to show vehicle leaving intersection
            if self.name == 'NORTH':
                vehicle.position = {'x': 0, 'y': -0.8}
            elif self.name == 'SOUTH':
                vehicle.position = {'x': 0, 'y': 0.8}
            elif self.name == 'EAST':
                vehicle.position = {'x': -0.8, 'y': 0}
            else:  # WEST
                vehicle.position = {'x': 0.8, 'y': 0}

            processed_vehicles.append(vehicle)
            processed += 1

        # Update remaining vehicles' positions
        self.update_vehicle_positions()

        return processed, processed_vehicles
    
    def get_priority_score(self, current_time):
        vehicle_count = len(self.vehicles)
        has_emergency = any(v.is_emergency for v in self.vehicles)
        time_since_green = current_time - self.last_green_time
        
        priority = (
            vehicle_count * 2 +
            (100 if has_emergency else 0) +
            min(time_since_green * 0.1, 10)
        )
        
        return priority, has_emergency

class VisualTrafficSimulation:
    def __init__(self):
        self.lanes = {
            'NORTH': VisualTrafficLane('NORTH', 'NS'),
            'SOUTH': VisualTrafficLane('SOUTH', 'NS'), 
            'EAST': VisualTrafficLane('EAST', 'EW'),
            'WEST': VisualTrafficLane('WEST', 'EW')
        }
        self.current_time = 0
        self.simulation_history = []
        self.processed_vehicles_animation = []  # For showing vehicles moving through
        
    def generate_vehicles(self):
        for lane_name, lane in self.lanes.items():
            if random.random() < 0.7:  # Increased to 70% chance for better visibility
                # Random vehicle type
                rand = random.random()
                if rand < 0.05:  # 5% emergency
                    vehicle_type = 'emergency'
                elif rand < 0.20:  # 15% trucks
                    vehicle_type = 'truck'
                else:  # 80% cars
                    vehicle_type = 'car'

                vehicle = Vehicle(lane_name, self.current_time, vehicle_type)
                lane.add_vehicle(vehicle)
    
    def priority_schedule(self):
        max_priority = -1
        selected_lane = None
        
        for lane_name, lane in self.lanes.items():
            if lane.vehicles:
                priority, _ = lane.get_priority_score(self.current_time)
                if priority > max_priority:
                    max_priority = priority
                    selected_lane = lane_name
                    
        return selected_lane
    
    def simulate_step(self, green_duration=30):
        self.current_time += 1
        self.generate_vehicles()
        
        selected_lane = self.priority_schedule()
        vehicles_processed = 0
        processed_vehicles = []
        
        if selected_lane and self.lanes[selected_lane].vehicles:
            vehicles_processed, processed_vehicles = self.lanes[selected_lane].process_vehicles(
                green_duration, self.current_time
            )
            
            self.lanes[selected_lane].current_signal = 'GREEN'
            self.lanes[selected_lane].last_green_time = self.current_time
            
            # Add processed vehicles to animation
            self.processed_vehicles_animation.extend(processed_vehicles)
            # Keep only recent processed vehicles for animation
            if len(self.processed_vehicles_animation) > 20:
                self.processed_vehicles_animation = self.processed_vehicles_animation[-20:]
            
            # Set other lanes to RED
            for lane_name, lane in self.lanes.items():
                if lane_name != selected_lane:
                    lane.current_signal = 'RED'
        
        # Calculate metrics
        lane_counts = {name: len(lane.vehicles) for name, lane in self.lanes.items()}
        wait_times = []
        for lane in self.lanes.values():
            for vehicle in lane.vehicles:
                wait_times.append(self.current_time - vehicle.arrival_time)
        
        avg_wait_time = np.mean(wait_times) if wait_times else 0
        
        step_data = {
            'time': self.current_time,
            'selected_lane': selected_lane,
            'vehicles_processed': vehicles_processed,
            'lane_counts': lane_counts.copy(),
            'avg_wait_time': avg_wait_time,
            'total_vehicles': sum(lane_counts.values())
        }
        
        self.simulation_history.append(step_data)
        return step_data

    def reset(self):
        """Reset the simulation to initial state"""
        self.__init__()

def create_visual_intersection(simulation):
    """Create detailed intersection with individual vehicles"""
    fig = go.Figure()
    
    # Draw intersection roads - make them narrower to avoid covering vehicles
    # Horizontal road
    fig.add_shape(type="rect", x0=-1.5, y0=-0.08, x1=1.5, y1=0.08,
                  fillcolor="#2c3e50", line_color="#34495e", line_width=2)
    # Vertical road
    fig.add_shape(type="rect", x0=-0.08, y0=-1.5, x1=0.08, y1=1.5,
                  fillcolor="#2c3e50", line_color="#34495e", line_width=2)
    
    # Add lane markings
    # Horizontal lane dividers
    fig.add_shape(type="line", x0=-1.5, y0=0, x1=-0.15, y1=0, 
                  line=dict(color="white", width=2, dash="dash"))
    fig.add_shape(type="line", x0=0.15, y0=0, x1=1.5, y1=0, 
                  line=dict(color="white", width=2, dash="dash"))
    # Vertical lane dividers
    fig.add_shape(type="line", x0=0, y0=-1.5, x1=0, y1=-0.15, 
                  line=dict(color="white", width=2, dash="dash"))
    fig.add_shape(type="line", x0=0, y0=0.15, x1=0, y1=1.5, 
                  line=dict(color="white", width=2, dash="dash"))
    
    # Add traffic lights at intersection corners
    traffic_light_positions = {
        'NORTH': (0.25, 0.25),
        'SOUTH': (-0.25, -0.25), 
        'EAST': (0.25, -0.25),
        'WEST': (-0.25, 0.25)
    }
    
    for lane_name, (x, y) in traffic_light_positions.items():
        lane = simulation.lanes[lane_name]
        color = '#27ae60' if lane.current_signal == 'GREEN' else '#e74c3c'
        
        # Traffic light pole
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=20, color=color, 
                       line=dict(width=3, color='black'),
                       symbol='square'),
            name=f'{lane_name}_light',
            showlegend=False
        ))
    
    # Draw individual vehicles in queues - ensure they appear on top
    all_vehicles_x = []
    all_vehicles_y = []
    all_vehicles_colors = []
    all_vehicles_sizes = []
    all_vehicles_symbols = []
    all_vehicles_text = []

    for lane_name, lane in simulation.lanes.items():
        for i, vehicle in enumerate(lane.vehicles):
            all_vehicles_x.append(vehicle.position['x'])
            all_vehicles_y.append(vehicle.position['y'])
            all_vehicles_colors.append(vehicle.color)
            all_vehicles_sizes.append(vehicle.size)

            # Different symbols for different vehicle types
            if vehicle.is_emergency:
                all_vehicles_symbols.append('diamond')
                all_vehicles_text.append('🚨')
            elif vehicle.vehicle_type == 'truck':
                all_vehicles_symbols.append('square')
                all_vehicles_text.append('🚛')
            else:
                all_vehicles_symbols.append('circle')
                all_vehicles_text.append('🚗')

    # Add all vehicles in one trace for better performance - make them prominent
    if all_vehicles_x:
        # Debug: Add a text annotation showing vehicle count
        fig.add_annotation(
            x=0.8, y=0.8, text=f"Vehicles: {len(all_vehicles_x)}",
            showarrow=False, font=dict(size=12, color='yellow'),
            bgcolor='rgba(0,0,0,0.8)', bordercolor='yellow', borderwidth=1
        )

        fig.add_trace(go.Scatter(
            x=all_vehicles_x, y=all_vehicles_y,
            mode='markers+text',
            marker=dict(size=all_vehicles_sizes, color=all_vehicles_colors,
                       line=dict(width=3, color='white'),
                       symbol=all_vehicles_symbols),
            text=all_vehicles_text,
            textposition='middle center',
            textfont=dict(size=10, color='white'),
            name='Vehicles',
            showlegend=False
        ))
    
    # Add processed vehicles animation (vehicles leaving intersection)
    if simulation.processed_vehicles_animation:
        processed_x = []
        processed_y = []

        for vehicle in simulation.processed_vehicles_animation[-3:]:  # Show last 3 processed
            # Position vehicles clearly outside the intersection
            if vehicle.lane == 'NORTH':
                processed_x.append(0)
                processed_y.append(-1.0)
            elif vehicle.lane == 'SOUTH':
                processed_x.append(0)
                processed_y.append(1.0)
            elif vehicle.lane == 'EAST':
                processed_x.append(-1.0)
                processed_y.append(0)
            else:  # WEST
                processed_x.append(1.0)
                processed_y.append(0)

        if processed_x:
            fig.add_trace(go.Scatter(
                x=processed_x, y=processed_y,
                mode='markers+text',
                marker=dict(size=12, color='#2ecc71',
                           line=dict(width=2, color='white')),
                text=['✅'] * len(processed_x),
                textposition='middle center',
                textfont=dict(size=8),
                name='Processed',
                showlegend=False
            ))
    
    # Add lane labels
    label_positions = {
        'NORTH ↑': (0, 1.3),
        'SOUTH ↓': (0, -1.3),
        'EAST →': (1.3, 0),
        '← WEST': (-1.3, 0)
    }
    
    for label, (x, y) in label_positions.items():
        fig.add_annotation(
            x=x, y=y, text=label,
            showarrow=False,
            font=dict(size=14, color='white'),
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='white',
            borderwidth=1
        )
    
    fig.update_layout(
        xaxis=dict(range=[-1.6, 1.6], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-1.6, 1.6], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='#1a252f',
        paper_bgcolor='#1a252f',
        height=500,
        title=dict(
            text="🚦 Live Traffic Intersection - Virtual Vehicles",
            font=dict(color='white', size=18),
            x=0.5
        )
    )
    
    return fig

# Streamlit App
st.set_page_config(
    page_title="GreenSched AI - Visual",
    page_icon="🚦",
    layout="wide"
)

# Initialize session state
if 'simulation' not in st.session_state:
    st.session_state.simulation = VisualTrafficSimulation()
    st.session_state.running = False

def main():
    simulation = st.session_state.simulation
    
    st.title("🚦 GreenSched AI - Visual Traffic Simulation")
    st.markdown("### Real-time Traffic with Virtual Vehicles 🚗🚛🚨")
    
    # Sidebar controls
    st.sidebar.header("🎛️ Simulation Controls")
    
    green_duration = st.sidebar.slider("Green Duration (seconds)", 10, 60, 30)
    
    # Vehicle spawn controls
    st.sidebar.subheader("🚗 Vehicle Generation")
    st.sidebar.write("Current spawn rate: 70% per lane per step")
    st.sidebar.write("Emergency vehicles: 5%")
    st.sidebar.write("Trucks: 15%")
    st.sidebar.write("Cars: 80%")
    
    # Control buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("▶️ Start"):
            st.session_state.running = True
    with col2:
        if st.button("⏸️ Pause"):
            st.session_state.running = False
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.simulation = VisualTrafficSimulation()
            st.session_state.running = False
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🚦 Live Intersection with Virtual Vehicles")
        intersection_fig = create_visual_intersection(simulation)
        st.plotly_chart(intersection_fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Live Statistics")
        st.metric("⏱️ Time", f"{simulation.current_time}s")
        
        if simulation.simulation_history:
            latest = simulation.simulation_history[-1]
            st.metric("🟢 Active Lane", latest['selected_lane'] or "None")
            st.metric("🚗 Total Vehicles", latest['total_vehicles'])
            st.metric("⏳ Avg Wait", f"{latest['avg_wait_time']:.1f}s")
            st.metric("✅ Processed", latest['vehicles_processed'])
        
        # Real-time queue status
        st.subheader("🚦 Queue Status")
        for lane_name, lane in simulation.lanes.items():
            signal_color = "🟢" if lane.current_signal == 'GREEN' else "🔴"
            count = len(lane.vehicles)
            
            # Count vehicle types
            cars = sum(1 for v in lane.vehicles if v.vehicle_type == 'car')
            trucks = sum(1 for v in lane.vehicles if v.vehicle_type == 'truck')
            emergency = sum(1 for v in lane.vehicles if v.is_emergency)
            
            vehicle_breakdown = f"🚗{cars}"
            if trucks > 0:
                vehicle_breakdown += f" 🚛{trucks}"
            if emergency > 0:
                vehicle_breakdown += f" 🚨{emergency}"
            
            st.write(f"{signal_color} **{lane_name}**: {count} vehicles ({vehicle_breakdown})")
    
    # Performance analytics
    if len(simulation.simulation_history) > 5:
        st.subheader("📈 Performance Analytics")
        
        df = pd.DataFrame(simulation.simulation_history)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.line(df, x='time', y='avg_wait_time', 
                          title='Average Wait Time Over Time',
                          color_discrete_sequence=['#3498db'])
            fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.line(df, x='time', y='vehicles_processed',
                          title='Vehicles Processed Per Cycle',
                          color_discrete_sequence=['#27ae60'])
            fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)
    
    # Legend
    with st.expander("🚦 Vehicle Legend"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("🚗 **Regular Cars** (Blue circles)")
            st.write("- 85% of traffic")
            st.write("- Standard processing time")
        with col2:
            st.write("🚛 **Trucks** (Red squares)")
            st.write("- 13% of traffic")  
            st.write("- Larger size, slower processing")
        with col3:
            st.write("🚨 **Emergency Vehicles** (Orange diamonds)")
            st.write("- 2% of traffic")
            st.write("- **Highest priority** - override normal scheduling")
    
    # Auto-run simulation
    if st.session_state.running:
        time.sleep(0.5)  # Faster for better vehicle flow visualization
        simulation.simulate_step(green_duration)
        st.rerun()

if __name__ == "__main__":
    main()