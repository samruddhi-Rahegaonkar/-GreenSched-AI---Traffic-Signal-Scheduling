import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
from collections import deque
import random
from sklearn.ensemble import RandomForestRegressor
import json
from datetime import datetime

# Import visual traffic simulation classes
from models import VisualTrafficSimulation, create_visual_intersection

def create_intersection_plot(simulation):
    """Create intersection visualization with FIXED textposition values"""
    fig = go.Figure()
    
    # Draw intersection roads
    fig.add_shape(type="rect", x0=-1, y0=-0.2, x1=1, y1=0.2, 
                  fillcolor="gray", line_color="white", line_width=2)
    fig.add_shape(type="rect", x0=-0.2, y0=-1, x1=0.2, y1=1, 
                  fillcolor="gray", line_color="white", line_width=2)
    
    # FIXED: Traffic light positions with correct textposition values
    positions = {
        'NORTH': (0, 0.8, 'bottom center'),
        'SOUTH': (0, -0.8, 'top center'), 
        'EAST': (0.8, 0, 'middle left'),
        'WEST': (-0.8, 0, 'middle right')
    }
    
    for lane_name, (x, y, textpos) in positions.items():
        lane = simulation.lanes[lane_name]
        vehicle_count = len(lane.vehicles)
        
        # Traffic light color
        color = 'green' if lane.current_signal == 'GREEN' else 'red'
        emergency = '🚨' if any(v.is_emergency for v in lane.vehicles) else ''
        
        # Add traffic light
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=30, color=color, line=dict(width=3, color='black')),
            text=f"{lane_name}<br>{vehicle_count} vehicles {emergency}",
            textposition=textpos,  # FIXED: Use correct textposition values
            textfont=dict(size=12, color='white'),
            name=lane_name,
            showlegend=False
        ))
    
    fig.update_layout(
        xaxis=dict(range=[-1.2, 1.2], showgrid=False, showticklabels=False),
        yaxis=dict(range=[-1.2, 1.2], showgrid=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        title="Real-time Intersection Status"
    )
    
    return fig

# Page config
st.set_page_config(
    page_title="GreenSched AI",
    page_icon="🚦",
    layout="wide"
)

# Initialize session state
if 'simulation' not in st.session_state:
    st.session_state.simulation = VisualTrafficSimulation()
    st.session_state.running = False

def main():
    simulation = st.session_state.simulation

    # Enhanced Header
    st.title("🚦 GreenSched AI - Enhanced")
    st.markdown("### Advanced Traffic Signal Scheduling with ML")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["🚦 Live Simulation", "📊 Analytics", "⚙️ Controls"])

    with tab1:
        live_simulation_tab(simulation)

    with tab2:
        analytics_tab(simulation)

    with tab3:
        controls_tab(simulation)

    # Get control parameters from controls tab (this will be called every time)
    # Note: In a real implementation, we'd need to handle this differently
    # For now, we'll use default values and the simulation will run with basic controls

    # Auto-run simulation with default parameters
    if st.session_state.running:
        # Use default parameters for now
        green_duration = 30
        simulation_speed = "Normal (1s)"

        # Parse simulation speed
        speed_map = {
            "Slow (2s)": 2.0,
            "Normal (1s)": 1.0,
            "Fast (0.5s)": 0.5,
            "Ultra Fast (0.1s)": 0.1
        }
        delay = speed_map.get(simulation_speed, 1.0)

        time.sleep(delay)
        simulation.simulate_step(green_duration)
        st.rerun()

def live_simulation_tab(simulation):
    """Enhanced live simulation view"""
    st.header("🚦 Live Traffic Intersection")

    # Real-time metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("⏱️ Time", f"{simulation.current_time}s")
    with col2:
        if simulation.simulation_history:
            latest = simulation.simulation_history[-1]
            st.metric("🚗 Total Vehicles", latest['total_vehicles'])
    with col3:
        if simulation.simulation_history:
            st.metric("⏳ Avg Wait", f"{latest['avg_wait_time']:.1f}s")
    with col4:
        active_lane = simulation.simulation_history[-1]['selected_lane'] if simulation.simulation_history else "None"
        st.metric("🟢 Active Lane", active_lane)

    # Enhanced intersection visualization
    intersection_fig = create_enhanced_intersection(simulation)
    st.plotly_chart(intersection_fig, use_container_width=True)

    # Live queue status with enhanced display
    st.subheader("🚦 Real-time Queue Status")
    cols = st.columns(4)
    for i, (lane_name, lane) in enumerate(simulation.lanes.items()):
        with cols[i]:
            signal_color = "🟢" if lane.current_signal == 'GREEN' else "🔴"
            count = len(lane.vehicles)
            emergency_count = sum(1 for v in lane.vehicles if v.is_emergency)

            st.markdown(f"""
            **{signal_color} {lane_name}**
            - Vehicles: {count}
            - Emergency: {emergency_count} 🚨
            - Status: {'Active' if lane.current_signal == 'GREEN' else 'Waiting'}
            """)

def analytics_tab(simulation):
    """Comprehensive analytics dashboard"""
    st.header("📊 Performance Analytics")

    if not simulation.simulation_history:
        st.info("Start the simulation to see analytics")
        return

    df = pd.DataFrame(simulation.simulation_history)

    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_wait = df['avg_wait_time'].mean()
        st.metric("Avg Wait Time", f"{avg_wait:.1f}", "seconds")
    with col2:
        total_processed = df['vehicles_processed'].sum()
        st.metric("Total Processed", total_processed)
    with col3:
        throughput = df['vehicles_processed'].mean()
        st.metric("Avg Throughput", f"{throughput:.1f}", "/cycle")
    with col4:
        max_vehicles = df['total_vehicles'].max()
        st.metric("Peak Vehicles", max_vehicles)

    # Performance Charts
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.line(df, x='time', y='avg_wait_time',
                      title='Average Wait Time Trend',
                      color_discrete_sequence=['#3498db'])
        fig1.update_layout(height=300)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.line(df, x='time', y='vehicles_processed',
                      title='Throughput Over Time',
                      color_discrete_sequence=['#27ae60'])
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Lane Performance Analysis
    st.subheader("📋 Lane Performance")
    lane_stats = {}
    for lane_name in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
        lane_data = df[df['selected_lane'] == lane_name]
        if not lane_data.empty:
            lane_stats[lane_name] = {
                'times_selected': len(lane_data),
                'avg_vehicles_processed': lane_data['vehicles_processed'].mean(),
                'total_vehicles_processed': lane_data['vehicles_processed'].sum()
            }

    if lane_stats:
        stats_df = pd.DataFrame.from_dict(lane_stats, orient='index')
        st.dataframe(stats_df)

def controls_tab(simulation):
    """Enhanced controls with presets and advanced options"""
    st.header("⚙️ Simulation Controls")

    # Scenario Presets
    st.subheader("🎭 Scenario Presets")
    scenario = st.selectbox(
        "Select Traffic Scenario",
        ["Normal Traffic", "Rush Hour", "Emergency Response", "Construction", "Custom"],
        help="Choose a preset scenario or customize parameters"
    )

    # Scenario-based parameter adjustment
    if scenario == "Rush Hour":
        default_duration = 45
        default_intensity = 1.5
        st.info("🚗 High traffic volume, longer green times")
    elif scenario == "Emergency Response":
        default_duration = 60
        default_intensity = 0.8
        st.info("🚨 Emergency vehicles prioritized, moderate traffic")
    elif scenario == "Construction":
        default_duration = 30
        default_intensity = 0.6
        st.info("🚧 Reduced traffic due to construction")
    else:
        default_duration = 30
        default_intensity = 1.0

    # Advanced Controls
    with st.expander("🔧 Advanced Parameters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            green_duration = st.slider(
                "Green Duration (seconds)",
                10, 120, default_duration,
                help="How long the green light stays active"
            )

            simulation_speed = st.selectbox(
                "Simulation Speed",
                ["Slow (2s)", "Normal (1s)", "Fast (0.5s)", "Ultra Fast (0.1s)"],
                index=1,
                help="Control how fast the simulation runs"
            )

        with col2:
            traffic_intensity = st.slider(
                "Traffic Intensity",
                0.1, 2.0, default_intensity, 0.1,
                help="Multiplier for vehicle spawn rate"
            )

            emergency_rate = st.slider(
                "Emergency Rate (%)",
                0, 20, 5,
                help="Percentage of emergency vehicles"
            )

    # Control Buttons with better layout
    st.subheader("🎮 Simulation Control")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            st.session_state.running = True
            st.success("Simulation started!")

    with col2:
        if st.button("⏸️ Pause", use_container_width=True):
            st.session_state.running = False
            st.info("Simulation paused")

    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            # Use reinitialization instead of reset method
            st.session_state.simulation = VisualTrafficSimulation()
            st.session_state.running = False
            st.warning("Simulation reset")

    with col4:
        if st.button("📊 Export Data", use_container_width=True):
            export_simulation_data(simulation)

    # Real-time parameter display
    st.subheader("📋 Current Parameters")
    param_col1, param_col2 = st.columns(2)

    with param_col1:
        st.write(f"**Scenario:** {scenario}")
        st.write(f"**Green Duration:** {green_duration}s")
        st.write(f"**Traffic Intensity:** {traffic_intensity}x")

    with param_col2:
        st.write(f"**Simulation Speed:** {simulation_speed}")
        st.write(f"**Emergency Rate:** {emergency_rate}%")
        st.write(f"**Status:** {'Running' if st.session_state.running else 'Stopped'}")

    return green_duration, simulation_speed, traffic_intensity, emergency_rate



def create_enhanced_intersection(simulation):
    """Enhanced intersection visualization with animations"""
    fig = go.Figure()

    # Enhanced road design
    fig.add_shape(type="rect", x0=-1.5, y0=-0.15, x1=1.5, y1=0.15,
                  fillcolor="#2c3e50", line_color="#34495e", line_width=3)
    fig.add_shape(type="rect", x0=-0.15, y0=-1.5, x1=0.15, y1=1.5,
                  fillcolor="#2c3e50", line_color="#34495e", line_width=3)

    # Lane markings and intersection box
    fig.add_shape(type="rect", x0=-0.15, y0=-0.15, x1=0.15, y1=0.15,
                  fillcolor="#34495e", line_color="white", line_width=2)

    # Draw vehicles with enhanced positioning
    all_vehicles_x = []
    all_vehicles_y = []
    all_vehicles_colors = []
    all_vehicles_sizes = []
    all_vehicles_text = []

    # Vehicle positioning logic (similar to original but enhanced)
    for lane_name, lane in simulation.lanes.items():
        for i, vehicle in enumerate(lane.vehicles):
            # Position vehicles in queues
            if lane_name == 'NORTH':
                x, y = 0, 0.3 + i * 0.08
            elif lane_name == 'SOUTH':
                x, y = 0, -0.3 - i * 0.08
            elif lane_name == 'EAST':
                x, y = 0.3 + i * 0.08, 0
            else:  # WEST
                x, y = -0.3 - i * 0.08, 0

            all_vehicles_x.append(x)
            all_vehicles_y.append(y)
            all_vehicles_colors.append(vehicle.color)
            all_vehicles_sizes.append(vehicle.size)
            # Fix: Use get_vehicle_emoji method if emoji attribute missing
            emoji = getattr(vehicle, 'emoji', vehicle.get_vehicle_emoji() if hasattr(vehicle, 'get_vehicle_emoji') else '🚗')
            all_vehicles_text.append(emoji)

    # Add vehicles to plot
    if all_vehicles_x:
        fig.add_trace(go.Scatter(
            x=all_vehicles_x, y=all_vehicles_y,
            mode='markers+text',
            marker=dict(size=all_vehicles_sizes, color=all_vehicles_colors,
                       line=dict(width=2, color='white')),
            text=all_vehicles_text,
            textposition='middle center',
            textfont=dict(size=10),
            name='Vehicles',
            showlegend=False
        ))

    # Traffic lights with enhanced design
    light_positions = {
        'NORTH': (0, 0.6), 'SOUTH': (0, -0.6),
        'EAST': (0.6, 0), 'WEST': (-0.6, 0)
    }

    for lane_name, (x, y) in light_positions.items():
        lane = simulation.lanes[lane_name]
        color = '#27ae60' if lane.current_signal == 'GREEN' else '#e74c3c'

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers',
            marker=dict(size=30, color=color, line=dict(width=3, color='black')),
            name=f'{lane_name}_light',
            showlegend=False
        ))

    # Enhanced layout
    fig.update_layout(
        xaxis=dict(range=[-1.8, 1.8], showgrid=False, showticklabels=False),
        yaxis=dict(range=[-1.8, 1.8], showgrid=False, showticklabels=False),
        plot_bgcolor='#1a252f',
        paper_bgcolor='#1a252f',
        height=500,
        title=dict(
            text="🚦 Enhanced Live Traffic Intersection",
            font=dict(color='white', size=16)
        )
    )

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

    return fig

def export_simulation_data(simulation):
    """Export simulation data to JSON"""
    if not simulation.simulation_history:
        st.error("No simulation data to export")
        return

    data = {
        'export_timestamp': datetime.now().isoformat(),
        'simulation_history': simulation.simulation_history,
        'total_steps': len(simulation.simulation_history),
        'final_metrics': simulation.simulation_history[-1] if simulation.simulation_history else {}
    }

    filename = f"greensched_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # Create download button
    json_data = json.dumps(data, indent=2, default=str)
    st.download_button(
        label="📥 Download Simulation Data",
        data=json_data,
        file_name=filename,
        mime="application/json"
    )

if __name__ == "__main__":
    main()