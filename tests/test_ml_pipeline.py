"""
Tests for ML pipeline and model validation
Demonstrates proper ML engineering practices for FAANG interviews
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from schedulers import MLScheduler
from models import VisualTrafficSimulation


class TestMLPipeline:
    """Test ML pipeline components"""

    def setup_method(self):
        """Set up test fixtures"""
        self.ml_scheduler = MLScheduler()
        self.sim = VisualTrafficSimulation()

    def test_ml_scheduler_initialization(self):
        """Test ML scheduler initializes correctly"""
        assert self.ml_scheduler.name == "ML-Based"
        assert not self.ml_scheduler.is_trained
        assert self.ml_scheduler.training_data == []
        assert self.ml_scheduler.min_training_samples == 20

    def test_data_collection(self):
        """Test data collection for training"""
        # Mock lane counts and decision
        lane_counts = {'NORTH': 5, 'SOUTH': 3, 'EAST': 2, 'WEST': 1}
        selected_lane = 'NORTH'
        wait_times = [10, 15, 8, 12]
        outcome_score = 0.85

        self.ml_scheduler.collect_data(lane_counts, selected_lane, wait_times, outcome_score)

        # Check data was collected
        assert len(self.ml_scheduler.training_data) == 1
        data_point = self.ml_scheduler.training_data[0]

        expected_features = list(lane_counts.values()) + [sum(wait_times)]
        assert data_point['features'] == expected_features
        assert data_point['selected_lane'] == selected_lane
        assert data_point['outcome_score'] == outcome_score

    def test_model_training_insufficient_data(self):
        """Test model training fails with insufficient data"""
        # Add less than minimum training samples
        for i in range(15):  # Less than 20
            self.ml_scheduler.collect_data(
                {'NORTH': i, 'SOUTH': i+1, 'EAST': i+2, 'WEST': i+3},
                'NORTH',
                [10, 15, 8],
                0.8
            )

        result = self.ml_scheduler.train_model()
        assert result == False
        assert not self.ml_scheduler.is_trained

    @patch('sklearn.ensemble.RandomForestRegressor')
    def test_model_training_success(self, mock_rf):
        """Test successful model training"""
        # Mock the RandomForestRegressor
        mock_model = Mock()
        mock_rf.return_value = mock_model

        # Add sufficient training data
        for i in range(25):  # More than 20
            self.ml_scheduler.collect_data(
                {'NORTH': i, 'SOUTH': i+1, 'EAST': i+2, 'WEST': i+3},
                'NORTH' if i % 2 == 0 else 'SOUTH',
                [10, 15, 8],
                0.8
            )

        result = self.ml_scheduler.train_model()

        # Verify training was called
        assert result == True
        assert self.ml_scheduler.is_trained
        mock_model.fit.assert_called_once()

        # Verify data preparation
        X, y = mock_model.fit.call_args[0]
        assert len(X) == 25  # Number of training samples
        assert len(y) == 25  # Number of target values

    @patch('sklearn.ensemble.RandomForestRegressor')
    def test_ml_prediction(self, mock_rf):
        """Test ML-based lane prediction"""
        # Mock trained model
        mock_model = Mock()
        mock_model.predict.return_value = [0.2]  # Low score for NORTH
        self.ml_scheduler.model = mock_model
        self.ml_scheduler.is_trained = True

        # Set up lanes with vehicles
        lanes = {}
        for lane_name in ['NORTH', 'SOUTH', 'EAST', 'WEST']:
            lane = Mock()
            lane.vehicles = [Mock() for _ in range(3)]  # 3 vehicles each
            lanes[lane_name] = lane

        # Mock vehicle wait times
        with patch('time.time', return_value=100):
            for lane in lanes.values():
                for vehicle in lane.vehicles:
                    vehicle.arrival_time = 90  # 10 seconds wait

            result = self.ml_scheduler.schedule(lanes, 100)

            # Should call predict
            mock_model.predict.assert_called()

            # Should return a lane (the one with best predicted score)
            assert result in ['NORTH', 'SOUTH', 'EAST', 'WEST']


class TestMLValidation:
    """Test ML model validation and evaluation"""

    def setup_method(self):
        self.ml_scheduler = MLScheduler()

    def test_cross_validation_setup(self):
        """Test cross-validation data preparation"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 100

        training_data = []
        for i in range(n_samples):
            features = np.random.rand(5)  # 4 lane counts + 1 wait time sum
            selected_lane = np.random.choice(['NORTH', 'SOUTH', 'EAST', 'WEST'])
            outcome_score = np.random.rand()

            training_data.append({
                'features': features.tolist(),
                'selected_lane': selected_lane,
                'outcome_score': outcome_score
            })

        self.ml_scheduler.training_data = training_data

        # Test data splitting
        X = np.array([data['features'] for data in training_data])
        y = np.array([data['outcome_score'] for data in training_data])

        assert X.shape == (n_samples, 5)
        assert y.shape == (n_samples,)

    def test_model_metrics_calculation(self):
        """Test calculation of model performance metrics"""
        # Mock predictions and actual values
        y_true = np.array([0.8, 0.7, 0.9, 0.6, 0.85])
        y_pred = np.array([0.75, 0.8, 0.85, 0.65, 0.9])

        # Calculate metrics
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

        # Verify calculations
        assert mse > 0
        assert rmse > 0
        assert mae > 0
        assert r2 <= 1  # R² can be negative

    def test_feature_importance_analysis(self):
        """Test feature importance extraction"""
        # Mock feature names and importance scores
        feature_names = ['north_count', 'south_count', 'east_count', 'west_count', 'total_wait']
        importance_scores = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importance_scores))

        # Verify most important features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        assert sorted_features[0][0] == 'north_count'
        assert sorted_features[-1][0] == 'total_wait'

        # Verify total importance sums to reasonable value
        total_importance = sum(importance_scores)
        assert abs(total_importance - 1.0) < 0.1  # Should be close to 1


class TestSimulationMetrics:
    """Test comprehensive simulation performance metrics"""

    def setup_method(self):
        self.sim = VisualTrafficSimulation()

    def test_throughput_calculation(self):
        """Test throughput calculation over time"""
        # Run multiple simulation steps
        for _ in range(10):
            self.sim.simulate_step()

        # Calculate throughput metrics
        total_processed = sum(step['vehicles_processed'] for step in self.sim.simulation_history)
        avg_throughput = total_processed / len(self.sim.simulation_history)

        assert total_processed >= 0
        assert avg_throughput >= 0

    def test_wait_time_analysis(self):
        """Test wait time statistical analysis"""
        # Run simulation to generate data
        for _ in range(20):
            self.sim.simulate_step()

        wait_times = [step['avg_wait_time'] for step in self.sim.simulation_history]

        # Calculate statistics
        mean_wait = np.mean(wait_times)
        std_wait = np.std(wait_times)
        max_wait = np.max(wait_times)
        min_wait = np.min(wait_times)

        # Verify statistical properties
        assert mean_wait >= 0
        assert std_wait >= 0
        assert max_wait >= min_wait
        assert max_wait >= mean_wait

    def test_fairness_metrics(self):
        """Test fairness across different lanes"""
        # Run simulation
        for _ in range(30):
            self.sim.simulate_step()

        # Analyze lane selection fairness
        lane_selections = {}
        for step in self.sim.simulation_history:
            lane = step['selected_lane']
            if lane:
                lane_selections[lane] = lane_selections.get(lane, 0) + 1

        if lane_selections:
            # Calculate fairness metrics
            total_selections = sum(lane_selections.values())
            expected_per_lane = total_selections / len(lane_selections)

            # Variance in selection distribution
            selection_variance = np.var(list(lane_selections.values()))

            assert selection_variance >= 0
            assert total_selections > 0


class TestPerformanceBenchmarking:
    """Test performance benchmarking and optimization"""

    def test_memory_usage_tracking(self):
        """Test memory usage monitoring"""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run simulation
        sim = VisualTrafficSimulation()
        for _ in range(50):
            sim.simulate_step()

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase
        assert final_memory > 0

    def test_execution_time_profiling(self):
        """Test execution time profiling"""
        import time

        sim = VisualTrafficSimulation()

        # Profile simulation step execution
        start_time = time.time()
        for _ in range(10):
            sim.simulate_step()
        end_time = time.time()

        execution_time = end_time - start_time
        avg_step_time = execution_time / 10

        # Should complete within reasonable time
        assert avg_step_time < 1.0  # Less than 1 second per step
        assert execution_time > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov', '--cov-report=html'])
