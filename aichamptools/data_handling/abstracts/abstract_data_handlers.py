from abc import ABC, abstractmethod
from typing import Union, List, Optional



class AbstractExperimentsDataHandler(ABC):

    @abstractmethod
    def load_experiments(self, experiment_ids: Optional[Union[int, List[int]]] = None):
        """Load multiple experiments, optionally filtered by IDs."""
        pass

    @abstractmethod
    def save_experiment(self, name: str, description: str, status: str, created_at: str, updated_at: Optional[str] = None):
        """Save a new experiment with all required fields."""
        pass

    @abstractmethod
    def update_experiment(self, experiment_id: int, name: Optional[str] = None, description: Optional[str] = None, status: Optional[str] = None, updated_at: Optional[str] = None):
        """Update an existing experiment with optional fields."""
        pass

    @abstractmethod
    def delete_experiment(self, experiment_id: int):
        """Delete an experiment."""
        pass



class AbstractMessagesTemplatesDataHandler(ABC):

    @abstractmethod
    def load_messages_templates(self, template_ids: Optional[Union[int, List[int]]] = None, experiment_id: Optional[int] = None):
        """Load multiple message templates, optionally filtered by IDs or associated experiment."""
        pass

    @abstractmethod
    def save_messages_template(self, messages: dict, description: str, created_at: str, updated_at: Optional[str] = None, experiment_id: Optional[int] = None):
        """Save a new messages template with all required fields."""
        pass

    @abstractmethod
    def update_messages_template(self, template_id: int, messages: Optional[dict] = None, description: Optional[str] = None, updated_at: Optional[str] = None, experiment_id: Optional[int] = None):
        """Update an existing messages template with optional fields."""
        pass

    @abstractmethod
    def delete_messages_template(self, template_id: int):
        """Delete a messages template."""
        pass



class AbstractDataPointsDataHandler(ABC):

    @abstractmethod
    def load_datapoints(self, datapoint_ids: Optional[Union[int, List[int]]] = None, experiment_id: Optional[int] = None):
        """Load multiple data points, optionally filtered by IDs or associated experiment."""
        pass

    @abstractmethod
    def save_datapoint(self, experiment_id: int, data: dict, description: str, created_at: str, updated_at: Optional[str] = None):
        """Save a new data point, linking it to an experiment with all required fields."""
        pass

    @abstractmethod
    def update_datapoint(self, data_point_id: int, experiment_id: Optional[int] = None, data: Optional[dict] = None, description: Optional[str] = None, updated_at: Optional[str] = None):
        """Update an existing data point with optional fields."""
        pass

    @abstractmethod
    def delete_datapoint(self, data_point_id: int):
        """Delete a data point."""
        pass



class AbstractCompletionsDataHandler(ABC):

    @abstractmethod
    def load_completions(self, completion_ids: Optional[Union[int, List[int]]] = None, generation_id: Optional[int] = None, experiment_id: Optional[int] = None):
        """Load multiple completions, optionally filtered by IDs or associated generation or experiment."""
        pass

    @abstractmethod
    def save_completion(self, generation_id: int, execution_params: dict, messages_template_id: int, data_point_id: int, messages: dict, completion: dict, type: str, created_at: str):
        """Save a new completion with all required fields."""
        pass

    @abstractmethod
    def update_completion(self, completion_id: int, execution_params: Optional[dict] = None, messages_template_id: Optional[int] = None, data_point_id: Optional[int] = None, messages: Optional[dict] = None, completion: Optional[dict] = None, type: Optional[str] = None):
        """Update an existing completion with optional fields."""
        pass

    @abstractmethod
    def delete_completion(self, completion_id: int):
        """Delete a completion."""
        pass



class AbstractGenerationsDataHandler(ABC):

    @abstractmethod
    def load_generations(self, generation_ids: Optional[Union[int, List[int]]] = None, experiment_id: Optional[int] = None):
        """Load multiple generations, optionally filtered by IDs or associated experiment."""
        pass

    @abstractmethod
    def save_generation(self, experiment_id: int, start_time: str, end_time: Optional[str], status: str):
        """Save a new generation with required fields."""
        pass

    @abstractmethod
    def update_generation(self, generation_id: int, start_time: Optional[str] = None, end_time: Optional[str] = None, status: Optional[str] = None):
        """Update an existing generation with optional fields."""
        pass

    @abstractmethod
    def delete_generation(self, generation_id: int):
        """Delete a generation."""
        pass



class AbstractAssessmentsDataHandler(ABC):

    @abstractmethod
    def load_assessments(self, assessment_ids: Optional[Union[int, List[int]]] = None, experiment_id: Optional[int] = None, target_completion_id: Optional[int] = None, judge_completion_id: Optional[int] = None):
        """Load multiple assessments, optionally filtered by IDs or associated completions or experiment."""
        pass

    @abstractmethod
    def save_assessment(self, target_completion_id: int, judge_completion_id: Optional[int], results: dict, created_at: str, experiment_id: Optional[int] = None):
        """Save a new assessment with all required fields."""
        pass

    @abstractmethod
    def update_assessment(self, assessment_id: int, target_completion_id: Optional[int] = None, judge_completion_id: Optional[int] = None, results: Optional[dict] = None):
        """Update an existing assessment with optional fields."""
        pass

    @abstractmethod
    def delete_assessment(self, assessment_id: int):
        """Delete an assessment."""
        pass

