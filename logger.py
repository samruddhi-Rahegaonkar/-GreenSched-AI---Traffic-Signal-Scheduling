"""
Logging configuration for GreenSched AI
Demonstrates production-quality logging for FAANG interviews
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level=logging.INFO, log_to_file=True):
    """
    Set up comprehensive logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file in addition to console
    """
    # Create logger
    logger = logging.getLogger('greensched_ai')
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if requested)
    if log_to_file:
        # Create logs directory
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # Create rotating file handler
        log_file = log_dir / f'greensched_ai_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5  # 10MB per file, 5 backups
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name='greensched_ai'):
    """
    Get a logger instance with the specified name

    Args:
        name: Logger name (will be prefixed with 'greensched_ai')

    Returns:
        Logger instance
    """
    return logging.getLogger(f'greensched_ai.{name}')


class SimulationLogger:
    """Specialized logger for simulation events"""

    def __init__(self):
        self.logger = get_logger('simulation')

    def log_simulation_start(self, config):
        """Log simulation start with configuration"""
        self.logger.info(f"Simulation started with config: {config}")

    def log_simulation_step(self, step_data):
        """Log simulation step completion"""
        self.logger.info(
            f"Step {step_data['time']}: Selected lane {step_data['selected_lane']}, "
            f"Processed {step_data['vehicles_processed']} vehicles, "
            f"Avg wait time: {step_data['avg_wait_time']:.2f}s"
        )

    def log_algorithm_selection(self, algorithm, reason=""):
        """Log algorithm selection"""
        self.logger.info(f"Selected algorithm: {algorithm} {reason}")

    def log_ml_training(self, samples, accuracy=None):
        """Log ML model training"""
        msg = f"ML model trained with {samples} samples"
        if accuracy:
            msg += f", accuracy: {accuracy:.3f}"
        self.logger.info(msg)

    def log_performance_metrics(self, metrics):
        """Log performance metrics"""
        self.logger.info(f"Performance metrics: {metrics}")

    def log_error(self, error, context=""):
        """Log errors with context"""
        self.logger.error(f"Error in {context}: {error}", exc_info=True)

    def log_warning(self, warning, context=""):
        """Log warnings with context"""
        self.logger.warning(f"Warning in {context}: {warning}")


class MLLogger:
    """Specialized logger for ML operations"""

    def __init__(self):
        self.logger = get_logger('ml')

    def log_data_collection(self, samples_count, features_count):
        """Log data collection progress"""
        self.logger.info(f"Collected {samples_count} samples with {features_count} features")

    def log_model_training_start(self, algorithm, params=None):
        """Log model training start"""
        msg = f"Starting {algorithm} model training"
        if params:
            msg += f" with params: {params}"
        self.logger.info(msg)

    def log_model_training_complete(self, algorithm, metrics):
        """Log model training completion"""
        self.logger.info(f"{algorithm} model training completed. Metrics: {metrics}")

    def log_prediction(self, features, prediction, confidence=None):
        """Log model prediction"""
        msg = f"Prediction: {prediction} for features: {features}"
        if confidence:
            msg += f" (confidence: {confidence:.3f})"
        self.logger.debug(msg)

    def log_feature_importance(self, features, importance_scores):
        """Log feature importance analysis"""
        self.logger.info("Feature importance analysis:")
        for feature, importance in zip(features, importance_scores):
            self.logger.info(f"  {feature}: {importance:.4f}")

    def log_model_validation(self, validation_results):
        """Log model validation results"""
        self.logger.info(f"Model validation results: {validation_results}")


# Global logger instances
simulation_logger = SimulationLogger()
ml_logger = MLLogger()

# Convenience functions
def log_simulation_event(event_type, **kwargs):
    """Convenience function for logging simulation events"""
    if event_type == 'start':
        simulation_logger.log_simulation_start(kwargs.get('config', {}))
    elif event_type == 'step':
        simulation_logger.log_simulation_step(kwargs.get('step_data', {}))
    elif event_type == 'algorithm':
        simulation_logger.log_algorithm_selection(
            kwargs.get('algorithm', ''),
            kwargs.get('reason', '')
        )
    elif event_type == 'error':
        simulation_logger.log_error(
            kwargs.get('error', ''),
            kwargs.get('context', '')
        )


def log_ml_event(event_type, **kwargs):
    """Convenience function for logging ML events"""
    if event_type == 'training_start':
        ml_logger.log_model_training_start(
            kwargs.get('algorithm', ''),
            kwargs.get('params', None)
        )
    elif event_type == 'training_complete':
        ml_logger.log_model_training_complete(
            kwargs.get('algorithm', ''),
            kwargs.get('metrics', {})
        )
    elif event_type == 'prediction':
        ml_logger.log_prediction(
            kwargs.get('features', []),
            kwargs.get('prediction', None),
            kwargs.get('confidence', None)
        )


# Initialize logging when module is imported
default_logger = setup_logging()
