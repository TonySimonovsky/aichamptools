# # __init__.py for AIChampTools package

# # This file initializes the AIChampTools package and exposes its main functionalities.

# # Import core functionalities
# from .core.base import BaseClass  # Adjust the import according to your actual class names and structure

# Import connectors
from .llms.anthropic import LLMAnthropic
from .llms.mistralai import LLMMistral
from .llms.openai import LLMOpenAI

# # Import assessment tools
# from .assessment.assessor import Assessor
# from .assessment.criteria import Criteria

# # Import utilities
# from .utilities.logging_utils import setup_logging, log_message
# from .utilities.api_utils import send_request, handle_api_error

# # Import data handling tools
# from .data_handling.data_processor import DataProcessor
# from .data_handling.data_storage import DataStorage

# __version__ = '0.1.0'

# __all__ = [
#     'BaseClass',
#     'OpenAIConnector', 'MistralConnector', 'AnthropicConnector',
#     'Assessor', 'Criteria',
#     'setup_logging', 'log_message',
#     'send_request', 'handle_api_error',
#     'DataProcessor', 'DataStorage'
# ]