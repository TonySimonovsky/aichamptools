# logging_utils.py

# This module manages logging across the LLMAssessMate application.
# logging_utils.py

import logging
import os
import inspect


# Global variable to control logging
logging_enabled = True


def setup_logging(logs_folder="logs_aichamptools/", log_filename="step_by_step.log", level=logging.INFO):
    """Set up logging configuration."""
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)

    log_path = os.path.join(logs_folder, log_filename)
    logger = logging.getLogger('step_by_step')
    logger.setLevel(level)

    # Remove all handlers associated with the logger object.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    return logger

def log_message(logger, level, class_instance, message, user_id=None):

    if logging_enabled:
        """Log a message with the given level on the provided logger."""
        current_frame = inspect.currentframe()
        frame_info = inspect.getframeinfo(current_frame.f_back)
        
        file_name = os.path.basename(frame_info.filename)  # Get only the base filename, not the full path
        line_number = frame_info.lineno
        class_name = class_instance.__class__.__name__
        func_name = current_frame.f_back.f_code.co_name

        # Check if the logging level is valid
        if level not in ['debug', 'info', 'warning', 'error', 'critical']:
            level = 'info'

        log_func = getattr(logger, level)
        log_message = f'{file_name}:{line_number} - {class_name} - {func_name} - {message}'

        # Add user ID to the log message if it's provided
        if user_id is not None:
            log_message += f' - user {user_id}'

        log_func(log_message)

def enable_logging(enable=True):
    """Enable or disable logging."""
    global logging_enabled
    logging_enabled = enable