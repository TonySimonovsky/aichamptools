from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, TIMESTAMP, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import json


Base = declarative_base()

class Experiment(Base):
    __tablename__ = 'experiments'
    experiment_id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    status = Column(String(50), nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)
    updated_at = Column(TIMESTAMP)

class Generation(Base):
    __tablename__ = 'generations'
    generation_id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'), nullable=False)
    start_time = Column(TIMESTAMP, nullable=False)
    end_time = Column(TIMESTAMP)
    status = Column(String(50), nullable=False)
    experiment = relationship("Experiment")

class Completion(Base):
    __tablename__ = 'completions'
    completion_id = Column(Integer, primary_key=True)
    generation_id = Column(Integer, ForeignKey('generations.generation_id'))
    execution_params = Column(JSON)
    messages_template_id = Column(Integer, ForeignKey('messages_templates.template_id'))
    datapoint_id = Column(Integer, ForeignKey('datapoints.datapoint_id'))
    messages = Column(JSON, nullable=False)
    completion = Column(JSON, nullable=False)
    type = Column(String(50), nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)
    generation = relationship("Generation")
    messages_template = relationship("MessagesTemplate")
    datapoint = relationship("DataPoint")

class Assessment(Base):
    __tablename__ = 'assessments'
    assessment_id = Column(Integer, primary_key=True)
    target_completion_id = Column(Integer, ForeignKey('completions.completion_id'), nullable=False)
    judge_completion_id = Column(Integer, ForeignKey('completions.completion_id'))
    results = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP, nullable=False)
    target_completion = relationship("Completion", foreign_keys=[target_completion_id])
    judge_completion = relationship("Completion", foreign_keys=[judge_completion_id])

class MessagesTemplate(Base):
    __tablename__ = 'messages_templates'
    template_id = Column(Integer, primary_key=True)
    messages_template = Column(JSON, nullable=False)
    description = Column(Text)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'), nullable=True)
    created_at = Column(TIMESTAMP, nullable=False)
    updated_at = Column(TIMESTAMP)

class DataPoint(Base):
    __tablename__ = 'datapoints'
    datapoint_id = Column(Integer, primary_key=True)
    experiment_id = Column(Integer, ForeignKey('experiments.experiment_id'), nullable=False)
    data = Column(JSON, nullable=False)
    description = Column(Text)
    created_at = Column(TIMESTAMP, nullable=False)
    updated_at = Column(TIMESTAMP)
    experiment = relationship("Experiment")



from typing import Optional, Union, List
from sqlalchemy.orm import Session
from .abstracts.abstract_data_handlers import *




class ExperimentsDataHandlerSQL(AbstractExperimentsDataHandler):
    def __init__(self, session: Session):
        self.session = session

    def load_experiments(self, experiment_ids: Optional[Union[int, List[int]]] = None):
        query = self.session.query(Experiment)
        if experiment_ids:
            query = query.filter(Experiment.experiment_id.in_(experiment_ids))
        return query.all()

    def save_experiment(self, name: str, description: str, status: str, created_at: str, updated_at: Optional[str] = None) -> dict:
        new_experiment = Experiment(name=name, description=description, status=status, created_at=created_at, updated_at=updated_at)
        self.session.add(new_experiment)
        self.session.commit()
        return {"experiment_id": new_experiment.experiment_id, "name": name, "description": description, "status": status, "created_at": created_at, "updated_at": updated_at}

    def update_experiment(self, experiment_id: int, name: Optional[str] = None, description: Optional[str] = None, status: Optional[str] = None, updated_at: Optional[str] = None):
        experiment = self.session.query(Experiment).get(experiment_id)
        if name:
            experiment.name = name
        if description:
            experiment.description = description
        if status:
            experiment.status = status
        if updated_at:
            experiment.updated_at = updated_at
        self.session.commit()

    def delete_experiment(self, experiment_id: int):
        experiment = self.session.query(Experiment).get(experiment_id)
        self.session.delete(experiment)
        self.session.commit()




class MessagesTemplatesDataHandlerSQL(AbstractMessagesTemplatesDataHandler):
    def __init__(self, session: Session):
        self.session = session

    def load_messages_templates(self, template_ids: Optional[Union[int, List[int]]] = None, experiment_id: Optional[int] = None):
        query = self.session.query(MessagesTemplate)
        if template_ids:
            if isinstance(template_ids, int):
                template_ids = [template_ids]
            query = query.filter(MessagesTemplate.template_id.in_(template_ids))
        if experiment_id:
            query = query.filter(MessagesTemplate.experiment_id == experiment_id)
        templates = query.all()


        # Convert templates to list of dicts and parse JSON fields
        messages_templates = []
        for template in templates:
            template_dict = {key: value for key, value in template.__dict__.items() if key != '_sa_instance_state'}
            # Assuming 'messages_template' is the key that contains JSON string that needs to be converted
            if 'messages_template' in template_dict and template_dict['messages_template']:
                template_dict['messages_template'] = json.loads(template_dict['messages_template'])
            messages_templates.append(template_dict)


        return messages_templates


    def save_messages_template(self, messages_template: dict, description: str, created_at: str, updated_at: Optional[str] = None, experiment_id: Optional[int] = None) -> dict:
        new_template = MessagesTemplate(messages_template=messages_template, description=description, created_at=created_at, updated_at=updated_at, experiment_id=experiment_id)
        self.session.add(new_template)
        self.session.commit()
        return {
            "template_id": new_template.template_id,
            "messages_template": messages_template,
            "description": description,
            "created_at": created_at,
            "updated_at": updated_at,
            "experiment_id": experiment_id
        }

    def update_messages_template(self, template_id: int, messages_template: Optional[dict] = None, description: Optional[str] = None, updated_at: Optional[str] = None, experiment_id: Optional[int] = None):
        template = self.session.query(MessagesTemplate).get(template_id)
        if messages_template:
            template.messages = messages_template
        if description:
            template.description = description
        if updated_at:
            template.updated_at = updated_at
        if experiment_id:
            template.experiment_id = experiment_id
        self.session.commit()

    def delete_messages_template(self, template_id: int):
        template = self.session.query(MessagesTemplate).get(template_id)
        self.session.delete(template)
        self.session.commit()



class DataPointsDataHandlerSQL(AbstractDataPointsDataHandler):
    def __init__(self, session: Session):
        self.session = session

    def load_datapoints(self, datapoint_ids: Optional[Union[int, List[int]]] = None, experiment_id: Optional[int] = None):
        query = self.session.query(DataPoint)
        if datapoint_ids:
            if isinstance(datapoint_ids, int):
                datapoint_ids = [datapoint_ids]
            query = query.filter(DataPoint.datapoint_id.in_(datapoint_ids))
        if experiment_id:
            query = query.filter(DataPoint.experiment_id == experiment_id)
        datapoints = query.all()

        # Convert datapoints to list of dicts and parse JSON fields
        result = []
        for datapoint in datapoints:
            datapoint_dict = {key: value for key, value in datapoint.__dict__.items() if key != '_sa_instance_state'}
            # Assuming 'data' is the key that contains JSON string that needs to be converted
            if 'data' in datapoint_dict and datapoint_dict['data']:
                datapoint_dict['data'] = json.loads(datapoint_dict['data'])
            result.append(datapoint_dict)

        return result


    def save_datapoint(self, experiment_id: int, data: dict, description: str, created_at: str, updated_at: Optional[str] = None) -> dict:
        new_datapoint = DataPoint(experiment_id=experiment_id, data=data, description=description, created_at=created_at, updated_at=updated_at)
        self.session.add(new_datapoint)
        self.session.commit()
        return {
            "datapoint_id": new_datapoint.datapoint_id,
            "experiment_id": experiment_id,
            "data": data,
            "description": description,
            "created_at": created_at,
            "updated_at": updated_at
        }

    def update_datapoint(self, datapoint_id: int, experiment_id: Optional[int] = None, data: Optional[dict] = None, description: Optional[str] = None, updated_at: Optional[str] = None):
        datapoint = self.session.query(DataPoint).get(datapoint_id)
        if experiment_id:
            datapoint.experiment_id = experiment_id
        if data:
            datapoint.data = data
        if description:
            datapoint.description = description
        if updated_at:
            datapoint.updated_at = updated_at
        self.session.commit()

    def delete_datapoint(self, datapoint_id: int):
        datapoint = self.session.query(DataPoint).get(datapoint_id)
        self.session.delete(datapoint)
        self.session.commit()



class CompletionsDataHandlerSQL(AbstractCompletionsDataHandler):
    def __init__(self, session: Session):
        self.session = session

    def load_completions(self, completion_ids: Optional[Union[int, List[int]]] = None, generation_id: Optional[int] = None, experiment_id: Optional[int] = None):
        query = self.session.query(Completion)
        if completion_ids:
            query = query.filter(Completion.completion_id.in_(completion_ids))
        if generation_id:
            query = query.filter(Completion.generation_id == generation_id)
        if experiment_id:
            query = query.filter(Completion.experiment_id == experiment_id)
        return query.all()

    def save_completion(self, generation_id: int, execution_params: dict, messages_template_id: int, datapoint_id: int, messages: dict, completion: dict, type: str, created_at: str):
        new_completion = Completion(generation_id=generation_id, execution_params=execution_params, messages_template_id=messages_template_id, datapoint_id=datapoint_id, messages=messages, completion=completion, type=type, created_at=created_at)
        self.session.add(new_completion)
        self.session.commit()

    def update_completion(self, completion_id: int, execution_params: Optional[dict] = None, messages_template_id: Optional[int] = None, datapoint_id: Optional[int] = None, messages: Optional[dict] = None, completion: Optional[dict] = None, type: Optional[str] = None):
        completion = self.session.query(Completion).get(completion_id)
        if execution_params:
            completion.execution_params = execution_params
        if messages_template_id:
            completion.messages_template_id = messages_template_id
        if datapoint_id:
            completion.datapoint_id = datapoint_id
        if messages:
            completion.messages = messages
        if completion:
            completion.completion = completion
        if type:
            completion.type = type
        self.session.commit()

    def delete_completion(self, completion_id: int):
        completion = self.session.query(Completion).get(completion_id)
        self.session.delete(completion)
        self.session.commit()



class GenerationsDataHandlerSQL(AbstractGenerationsDataHandler):
    def __init__(self, session: Session):
        self.session = session

    def load_generations(self, generation_ids: Optional[Union[int, List[int]]] = None, experiment_id: Optional[int] = None):
        query = self.session.query(Generation)
        if generation_ids:
            query = query.filter(Generation.generation_id.in_(generation_ids))
        if experiment_id:
            query = query.filter(Generation.experiment_id == experiment_id)
        return query.all()

    # def save_generation(self, experiment_id: int, start_time: str, end_time: Optional[str], status: str):
    #     new_generation = Generation(experiment_id=experiment_id, start_time=start_time, end_time=end_time, status=status)
    #     self.session.add(new_generation)
    #     self.session.commit()


    def save_generation(self, experiment_id: int, completions: List[dict], start_time: Optional[str] = None, end_time: Optional[str] = None, status: Optional[str] = None):
        from datetime import datetime

        # Set default values if not provided
        if start_time is None:
            start_time = datetime.now().isoformat()  # Default to current time in ISO format
        if status is None:
            status = 'pending'  # Default status

        new_generation = Generation(experiment_id=experiment_id, start_time=start_time, end_time=end_time, status=status)
        self.session.add(new_generation)
        self.session.flush()  # Ensures 'new_generation' is assigned an ID before we use it

        # Save completions associated with this generation
        for completion in completions:
            new_completion = Completion(
                generation_id=new_generation.generation_id,
                execution_params=completion.get('execution_params', {}),
                messages_template_id=completion["messages_template"]["template_id"],
                datapoint_id=completion["datapoint"]['datapoint_id'],
                messages=completion["completion"]['messages'],
                completion=completion.get('completion', {}),
                type=completion.get('type', 'default'),
                created_at=completion["completion"]['created_at']
            )
            self.session.add(new_completion)

        self.session.commit()



    def update_generation(self, generation_id: int, start_time: Optional[str] = None, end_time: Optional[str] = None, status: Optional[str] = None):
        generation = self.session.query(Generation).get(generation_id)
        if start_time:
            generation.start_time = start_time
        if end_time:
            generation.end_time = end_time
        if status:
            generation.status = status
        self.session.commit()

    def delete_generation(self, generation_id: int):
        generation = self.session.query(Generation).get(generation_id)
        self.session.delete(generation)
        self.session.commit()



class AssessmentsDataHandlerSQL(AbstractAssessmentsDataHandler):
    def __init__(self, session: Session):
        self.session = session

    def load_assessments(self, assessment_ids: Optional[Union[int, List[int]]] = None, experiment_id: Optional[int] = None, target_completion_id: Optional[int] = None, judge_completion_id: Optional[int] = None):
        query = self.session.query(Assessment)
        if assessment_ids:
            query = query.filter(Assessment.assessment_id.in_(assessment_ids))
        if experiment_id:
            query = query.filter(Assessment.experiment_id == experiment_id)
        if target_completion_id:
            query = query.filter(Assessment.target_completion_id == target_completion_id)
        if judge_completion_id:
            query = query.filter(Assessment.judge_completion_id == judge_completion_id)
        return query.all()

    def save_assessment(self, target_completion_id: int, judge_completion_id: Optional[int], results: dict, created_at: str, experiment_id: Optional[int] = None):
        new_assessment = Assessment(target_completion_id=target_completion_id, judge_completion_id=judge_completion_id, results=results, created_at=created_at, experiment_id=experiment_id)
        self.session.add(new_assessment)
        self.session.commit()

    def update_assessment(self, assessment_id: int, target_completion_id: Optional[int] = None, judge_completion_id: Optional[int] = None, results: Optional[dict] = None):
        assessment = self.session.query(Assessment).get(assessment_id)
        if target_completion_id:
            assessment.target_completion_id = target_completion_id
        if judge_completion_id:
            assessment.judge_completion_id = judge_completion_id
        if results:
            assessment.results = results
        self.session.commit()

    def delete_assessment(self, assessment_id: int):
        assessment = self.session.query(Assessment).get(assessment_id)
        self.session.delete(assessment)
        self.session.commit()



