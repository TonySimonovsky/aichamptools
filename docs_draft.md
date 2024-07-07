AIChampTools -- a collection of tools for LLMOps

AIChampTools/
│
├── aichamptools/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py  # Base classes and core functionalities
│   │
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── openai_connector.py
│   │   ├── mistral_connector.py
│   │   ├── anthropic_connector.py
│   │
│   ├── assessment/
│   │   ├── __init__.py
│   │   ├── assessor.py
│   │   ├── criteria.py
│   │
│   ├── utilities/
│   │   ├── __init__.py
│   │   ├── logging_utils.py
│   │   ├── api_utils.py
│   │
│   └── data_handling/
│       ├── __init__.py
│       ├── data_processor.py
│       ├── data_storage.py