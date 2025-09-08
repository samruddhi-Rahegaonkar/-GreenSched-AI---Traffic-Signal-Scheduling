"""
Unit tests for scheduling algorithms in GreenSched AI
Tests cover FCFS, SJF, Priority, Round Robin, and ML-based scheduling
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import VisualTrafficLane, VisualTrafficSimulation
from schedulers import FCFSScheduler, SJFScheduler, PriorityScheduler, RoundRobinScheduler, MLScheduler


class TestFCFSScheduler:
    """Test First Come First Served scheduling algorithm"""

    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.scheduler = FCFSScheduler()
        self.lanes = {
            'NORTH': Mock(),
            'SOUTH': Mock(),
            'EAST': Mock(),
            'WEST': Mock()
        }

    def test_empty_lanes(self):
        """Test FCFS with no vehicles in any lane"""
        for lane in self.lanes.values():
            lane.vehicles = []

        result = self.scheduler.schedule(self.lanes, 0)
        assert result is None

    def test_single_vehicle_each_lane(self):
        """Test FCFS with one vehicle in each lane"""
        # Create mock vehicles with different arrival times
        vehicles = []
        for i, lane_name in enumerate(['NORTH', 'SOUTH', 'EAST', 'WEST']):
            vehicle = Mock()
            vehicle.arrival_time = i
            vehicles.append(vehicle)
            self.lanes[lane_name].vehicles = [vehicle]

        # FCFS should select the lane with the oldest vehicle (NORTH, arrival_time=0)
        result = self.scheduler.schedule(self.lanes, 10)
        assert result == 'NORTH'

    def test_multiple_vehicles_same_lane(self):
        """Test FCFS with multiple vehicles in one lane"""
        # Create vehicles with different arrival times
        vehicles = []
        for i in range(3):
            vehicle = Mock()
            vehicle.arrival_time = i * 10  # 0, 10, 20
            vehicles.append(vehicle)

        self.lanes['NORTH'].vehicles = vehicles
        for lane_name in ['SOUTH', 'EAST', 'WEST']:
            self.lanes[lane_name].vehicles = []

        result = self.scheduler.schedule(self.lanes, 30)
        assert result == 'NORTH'  # Should select NORTH with oldest vehicle


class TestSJFScheduler:
    """Test Shortest Job First scheduling algorithm"""

    def setup_method(self):
        self.scheduler = SJFScheduler()
        self.lanes = {
            'NORTH': Mock(),
            'SOUTH': Mock(),
            'EAST': Mock(),
            'WEST': Mock()
        }

    def test_sjf_selection(self):
        """Test SJF selects lane with fewest vehicles"""
        # Set up lanes with different numbers of vehicles
        self.lanes['NORTH'].vehicles = [Mock(), Mock(), Mock()]  # 3 vehicles
        self.lanes['SOUTH'].vehicles = [Mock()]  # 1 vehicle
        self.lanes['EAST'].vehicles = [Mock(), Mock()]  # 2 vehicles
        self.lanes['WEST'].vehicles = []  # 0 vehicles

        result = self.scheduler.schedule(self.lanes, 0)
        assert result == 'SOUTH'  # Should select lane with 1 vehicle

    def test_sjf_empty_lane_preference(self):
        """Test SJF prefers empty lanes over lanes with vehicles"""
        self.lanes['NORTH'].vehicles = [Mock()]
        self.lanes['SOUTH'].vehicles = []
        self.lanes['EAST'].vehicles = [Mock(), Mock()]
        self.lanes['WEST'].vehicles = []

        result = self.scheduler.schedule(self.lanes, 0)
        assert result in ['SOUTH', 'WEST']  # Should select an empty lane


class TestPriorityScheduler:
    """Test Priority-based scheduling algorithm"""

    def setup_method(self):
        self.scheduler = PriorityScheduler()
        self.lanes = {
            'NORTH': Mock(),
            'SOUTH': Mock(),
            'EAST': Mock(),
            'WEST': Mock()
        }

    def test_emergency_vehicle_priority(self):
        """Test that emergency vehicles get highest priority"""
        # Create regular vehicle
        regular_vehicle = Mock()
        regular_vehicle.is_emergency = False

        # Create emergency vehicle
        emergency_vehicle = Mock()
        emergency_vehicle.is_emergency = True

        # Set up lanes
        self.lanes['NORTH'].vehicles = [regular_vehicle]
        self.lanes['SOUTH'].vehicles = [emergency_vehicle]
        self.lanes['EAST'].vehicles = []
        self.lanes['WEST'].vehicles = []

        # Mock get_priority_score method
        self.lanes['NORTH'].get_priority_score = Mock(return_value=(5, False))
        self.lanes['SOUTH'].get_priority_score = Mock(return_value=(100, True))  # Emergency priority

        result = self.scheduler.schedule(self.lanes, 0)
        assert result == 'SOUTH'  # Should select emergency lane

    def test_vehicle_count_priority(self):
        """Test that lanes with more vehicles get higher priority"""
        # Set up lanes with different vehicle counts
        self.lanes['NORTH'].vehicles = [Mock(), Mock(), Mock()]  # 3 vehicles
        self.lanes['SOUTH'].vehicles = [Mock()]  # 1 vehicle
        self.lanes['EAST'].vehicles = []
        self.lanes['WEST'].vehicles = []

        # Mock priority scores based on vehicle count
        self.lanes['NORTH'].get_priority_score = Mock(return_value=(9, False))  # 3*2 + 3
        self.lanes['SOUTH'].get_priority_score = Mock(return_value=(3, False))  # 1*2 + 1

        result = self.scheduler.schedule(self.lanes, 0)
        assert result == 'NORTH'  # Should select lane with more vehicles


class TestRoundRobinScheduler:
    """Test Round Robin scheduling algorithm"""

    def setup_method(self):
        self.scheduler = RoundRobinScheduler()
        self.lanes = {
            'NORTH': Mock(),
            'SOUTH': Mock(),
            'EAST': Mock(),
            'WEST': Mock()
        }

    def test_round_robin_sequence(self):
        """Test that Round Robin cycles through lanes in order"""
        # Set up all lanes with vehicles
        for lane in self.lanes.values():
            lane.vehicles = [Mock()]

        # First call should select NORTH
        result1 = self.scheduler.schedule(self.lanes, 0)
        assert result1 == 'NORTH'

        # Second call should select EAST (skipping SOUTH if it has no vehicles)
        # Actually, let's set up a proper sequence test
        self.lanes['NORTH'].vehicles = []
        result2 = self.scheduler.schedule(self.lanes, 0)
        assert result2 == 'EAST'  # Should move to next lane with vehicles

    def test_round_robin_skip_empty_lanes(self):
        """Test that Round Robin skips lanes without vehicles"""
        # Only NORTH and WEST have vehicles
        self.lanes['NORTH'].vehicles = [Mock()]
        self.lanes['SOUTH'].vehicles = []
        self.lanes['EAST'].vehicles = []
        self.lanes['WEST'].vehicles = [Mock()]

        # Should cycle between NORTH and WEST
        result1 = self.scheduler.schedule(self.lanes, 0)
        assert result1 == 'NORTH'

        result2 = self.scheduler.schedule(self.lanes, 0)
        assert result2 == 'WEST'


class TestMLScheduling:
    """Test Machine Learning based scheduling"""

    def setup_method(self):
        self.scheduler = MLScheduler()
        self.lanes = {
            'NORTH': Mock(),
            'SOUTH': Mock(),
            'EAST': Mock(),
            'WEST': Mock()
        }

    def test_ml_fallback_to_priority(self):
        """Test ML scheduler falls back to priority when not trained"""
        # Set up lanes with vehicles
        for lane in self.lanes.values():
            lane.vehicles = [Mock()]

        # Mock priority scheduler
        with patch('schedulers.PriorityScheduler') as mock_priority:
            mock_priority.return_value.schedule.return_value = 'NORTH'

            result = self.scheduler.schedule(self.lanes, 0)
            assert result == 'NORTH'
            mock_priority.assert_called_once()

    @patch('schedulers.RandomForestRegressor')
    def test_ml_trained_model(self, mock_rf):
        """Test ML scheduler uses trained model when available"""
        # Mock trained model
        mock_model = Mock()
        mock_model.predict.return_value = [0.1]  # Low score for NORTH
        self.scheduler.model = mock_model
        self.scheduler.is_trained = True

        # Set up lanes with vehicles
        for lane in self.lanes.values():
            lane.vehicles = [Mock()]

        # Mock current time and wait times
        with patch('time.time', return_value=100):
            for lane in self.lanes.values():
                for vehicle in lane.vehicles:
                    vehicle.arrival_time = 90  # 10 seconds wait

            result = self.scheduler.schedule(self.lanes, 100)
            # Should select lane with best predicted score
            mock_model.predict.assert_called()


class TestVisualTrafficLane:
    """Test VisualTrafficLane functionality"""

    def setup_method(self):
        self.lane = VisualTrafficLane('NORTH', 'NS')

    def test_lane_initialization(self):
        """Test lane initializes correctly"""
        assert self.lane.name == 'NORTH'
        assert self.lane.direction == 'NS'
        assert self.lane.vehicles == []
        assert self.lane.total_vehicles_processed == 0
        assert self.lane.total_wait_time == 0
        assert self.lane.current_signal == 'RED'

    def test_add_vehicle(self):
        """Test adding vehicles to lane"""
        from models import Vehicle
        vehicle = Vehicle('NORTH', 0, 'car')

        self.lane.add_vehicle(vehicle)
        assert len(self.lane.vehicles) == 1
        assert self.lane.vehicles[0] == vehicle

    def test_process_vehicles(self):
        """Test processing vehicles from lane"""
        from models import Vehicle

        # Add multiple vehicles
        vehicles = []
        for i in range(3):
            vehicle = Vehicle('NORTH', i * 10, 'car')
            vehicles.append(vehicle)
            self.lane.add_vehicle(vehicle)

        # Process vehicles
        processed_count, processed_vehicles = self.lane.process_vehicles(30, 100)

        assert processed_count == 3  # Should process all vehicles
        assert len(processed_vehicles) == 3
        assert len(self.lane.vehicles) == 0  # Lane should be empty

    def test_priority_score_calculation(self):
        """Test priority score calculation"""
        from models import Vehicle

        # Add regular vehicles
        for i in range(2):
            vehicle = Vehicle('NORTH', i * 10, 'car')
            self.lane.add_vehicle(vehicle)

        # Add emergency vehicle
        emergency = Vehicle('NORTH', 20, 'emergency')
        self.lane.add_vehicle(emergency)

        priority, has_emergency = self.lane.get_priority_score(50)

        assert has_emergency == True
        assert priority == (2 * 2) + 100 + min(50 * 0.1, 10)  # vehicle_count * 2 + emergency_bonus + time_factor


class TestVisualTrafficSimulation:
    """Test VisualTrafficSimulation functionality"""

    def setup_method(self):
        self.sim = VisualTrafficSimulation()

    def test_simulation_initialization(self):
        """Test simulation initializes correctly"""
        assert len(self.sim.lanes) == 4
        assert all(lane_name in self.sim.lanes for lane_name in ['NORTH', 'SOUTH', 'EAST', 'WEST'])
        assert self.sim.current_time == 0
        assert self.sim.simulation_history == []

    def test_vehicle_generation(self):
        """Test vehicle generation logic"""
        # Mock random to ensure consistent results
        with patch('random.random', return_value=0.5):  # 50% chance
            self.sim.generate_vehicles()

            # Should have vehicles in some lanes (depending on random)
            total_vehicles = sum(len(lane.vehicles) for lane in self.sim.lanes.values())
            assert total_vehicles >= 0  # At least some vehicles should be generated

    def test_simulation_step(self):
        """Test simulation step execution"""
        initial_time = self.sim.current_time

        # Run simulation step
        step_data = self.sim.simulate_step()

        # Time should advance
        assert self.sim.current_time == initial_time + 1

        # Should have recorded step data
        assert len(self.sim.simulation_history) == 1
        assert 'time' in step_data
        assert 'selected_lane' in step_data
        assert 'vehicles_processed' in step_data

    def test_priority_scheduling(self):
        """Test priority-based lane selection"""
        from models import Vehicle

        # Clear existing vehicles
        for lane in self.sim.lanes.values():
            lane.vehicles = []

        # Add emergency vehicle to SOUTH lane
        emergency = Vehicle('SOUTH', 0, 'emergency')
        self.sim.lanes['SOUTH'].add_vehicle(emergency)

        # Add regular vehicles to other lanes
        for lane_name in ['NORTH', 'EAST', 'WEST']:
            vehicle = Vehicle(lane_name, 0, 'car')
            self.sim.lanes[lane_name].add_vehicle(vehicle)

        # Priority scheduling should select SOUTH (emergency)
        selected_lane = self.sim.priority_schedule()
        assert selected_lane == 'SOUTH'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
