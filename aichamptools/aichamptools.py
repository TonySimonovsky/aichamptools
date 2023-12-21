# import pandas as pd
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_colwidth', None)

# import os, sys
# import time
# import json

# from functools import reduce
# import operator

# from pydantic import BaseModel, Field, ValidationError
# from typing import Any, Dict, Optional, Tuple, ForwardRef, List, Union
# from enum import Enum

# from openai import OpenAI, ChatCompletion, APITimeoutError
# import tiktoken

# from datetime import datetime

# import copy

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# from typing import Any, Dict, Optional, Tuple, ForwardRef, List, Union
# from enum import Enum
# from pydantic import BaseModel, Field




# import logging
# import inspect


# # Create a logger for step-by-step logs
# sbs_logger = logging.getLogger('step_by_step')
# sbs_logger.setLevel(logging.INFO)  # Or whatever level you want

# # Remove all handlers associated with the logger object.
# for handler in sbs_logger.handlers[:]:
#     sbs_logger.removeHandler(handler)

# # Create a file handler
# sbs_handler = logging.FileHandler(f'step_by_step.log')
# sbs_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
# sbs_logger.addHandler(sbs_handler)

# def log(level, class_instance, message, user_id=None):
#     current_frame = inspect.currentframe()
#     frame_info = inspect.getframeinfo(current_frame.f_back)
    
#     file_name = os.path.basename(frame_info.filename)  # Get only the base filename, not the full path
#     line_number = frame_info.lineno
#     class_name = class_instance.__class__.__name__
#     func_name = current_frame.f_back.f_code.co_name

#     # Check if the logging level is valid
#     if level not in ['debug', 'info', 'warning', 'error', 'critical']:
#         level = 'info'

#     log_func = getattr(sbs_logger, level)
#     log_message = f'{file_name}:{line_number} - {class_name} - {func_name} - {message}'

#     # Add user ID to the log message if it's provided
#     if user_id is not None:
#         log_message += f' - user {user_id}'

#     log_func(log_message)



class AIChampTools():
    def __init__(self):
        print(self)




# class LLMUsage():

#     def __init__(self, expected_prompt_tokens:int=0, prompt_tokens:int=0, expected_completion_tokens:int=0, completion_tokens:int=0, expected_total_tokens:int=0, total_tokens:int=0, expected_total_cost:float=0.0, total_cost:float=0.0, generation_time:float=0.0):
#         # Expected number of prompt tokens
#         self.expected_prompt_tokens = expected_prompt_tokens
#         # Prompt tokens number received from LLM
#         self.prompt_tokens = prompt_tokens
#         self.expected_completion_tokens = expected_completion_tokens
#         self.completion_tokens = completion_tokens
#         self.expected_total_tokens = expected_total_tokens
#         self.total_tokens = total_tokens
#         self.expected_total_cost = expected_total_cost
#         self.total_cost = total_cost
#         self.generation_time = generation_time

#     def __add__(self, other):
#         if isinstance(other, LLMUsage):
#             return LLMUsage(
#                 self.expected_prompt_tokens + other.expected_prompt_tokens,
#                 self.prompt_tokens + other.prompt_tokens,
#                 self.expected_completion_tokens + other.expected_completion_tokens,
#                 self.completion_tokens + other.completion_tokens,
#                 self.expected_total_tokens + other.expected_total_tokens,
#                 self.total_tokens + other.total_tokens,
#                 self.expected_total_cost + other.expected_total_cost,
#                 self.total_cost + other.total_cost,
#                 self.generation_time + other.generation_time
#             )
#         else:
#             raise TypeError("Unsupported operand type. Both operands should be instances of LLMUsage.")



# class LLM():

#     models = {}

#     def __init__(self):
#         # a flag if the LLM requires a user message to get completion (for example, in OpenAI you can send only system message, but Mistral will throw an error if you do so with a user's message)
#         self.requires_user_message = False

#     def execution_cost(self, model:str, llm_usage:LLMUsage) -> float:

#         sbs_logger.info(f"START")
#         sbs_logger.info(f"model: {model}")
#         sbs_logger.info(f"llm_usage: {llm_usage}")

#         if model in self.models:
#             pricing = self.models[model]["pricing"]
#         else:
#             pricing = self.models[model]["no-pricing"]

#         cost = (llm_usage.prompt_tokens/1000)*pricing['prompt_tokens'] + (llm_usage.completion_tokens/1000)*pricing['completion_tokens']

#         return cost




# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage

# class LLMMistral(LLM):

#     vendor="MistralAI"


#     models = {
#         # from https://docs.mistral.ai/platform/pricing/ on 2023.12.15
#         "mistral-tiny": { "pricing": {"prompt_tokens": 0.14*1.1/1000, "completion_tokens": 0.42*1.1/1000}},
#         "mistral-small": { "pricing": {"prompt_tokens": 0.6*1.1/1000, "completion_tokens": 1.8*1.1/1000}},
#         "mistral-medium": { "pricing": {"prompt_tokens": 2.5*1.1/1000, "completion_tokens": 7.5*1.1/1000}},
#         "no-pricing": { "pricing": {"prompt_tokens": 0, "completion_tokens": 0}}
#     }

#     def __init__(self, api_key=os.environ["MISTRAL_API_KEY"]):
#         super().__init__()  # Call the parent class's __init__ method

#         self.api_key = api_key
#         self.client = MistralClient(api_key=self.api_key)
#         self.requires_user_message = True
    

#     def create_completion(self, llm_params, messages):

#         llm_params_copy = copy.deepcopy(llm_params)

#         log("info", self, f"""START""")
#         log("info", self, f"""INPUT: llm_params: {llm_params_copy}""")
#         log("info", self, f"""INPUT: messages: {json.dumps(messages,indent=4)}""")

#         # checking of there the last message is from the user (Mistral LLM requirement)
#         if messages[-1]["role"] != "user":
#             messages.append({"role": "user", "content":"[ignore this message and continue following your instructions]"})

#         try:
#             messages = [
#                 ChatMessage(role=m["role"], content=m["content"]) for m in messages
#             ]
#         except Exception as e:
#             log("error", self, f"""Couldn't convert to ChatMessage objects: {e}""")


#         reps = llm_params_copy.pop("n", None) or 1

#         llm_responses_all = { "choices": [], "usage": { "prompt_tokens": 0, "completion_tokens": 0, "generation_time": 0 } }
#         for i in range(reps):

#             log("info", self, f"""rep {i+1}/{reps}""")

#             llm_response = None
#             llm_generation_time = 0

#             try:
#                 start_time = time.time()
#                 llm_response = self.client.chat(
#                     model=llm_params_copy["model"],
#                     messages=messages
#                 )
#                 end_time = time.time()
#                 llm_generation_time = end_time - start_time
#             except Exception as e:
#                 log("error", self, f"""Error while trying to generate a completion: {e}""")

#             llm_response = llm_response.model_dump()

#             log("info", self, f"""llm_response {i+1}/{reps}: {llm_response}""")

#             llm_response["usage"]["generation_time"] = llm_generation_time

#             llm_responses_all["choices"].append(llm_response["choices"][0])

#             llm_responses_all["usage"]["prompt_tokens"] = llm_responses_all["usage"].get("prompt_tokens") or llm_response["usage"]["prompt_tokens"]
#             llm_responses_all["usage"]["completion_tokens"] += llm_responses_all["usage"]["completion_tokens"]
#             llm_responses_all["usage"]["generation_time"] += llm_generation_time


#         log("info", self, f"""RETURNING: {json.dumps(llm_responses_all,indent=4,default=str)}""")

#         return llm_responses_all



# class LLMOpenAI(LLM):
#     vendor="OpenAI"

#     models = {
#         # from https://openai.com/pricing on 2023.11.16
#         "gpt-4-1106-preview": { "pricing": {"prompt_tokens": 0.01, "completion_tokens": 0.03}},
#         "gpt-4-1106-vision-preview": { "pricing": {"prompt_tokens": 0.01, "completion_tokens": 0.03}},
#         "gpt-4": { "pricing": {"prompt_tokens": 0.03, "completion_tokens": 0.06}},
#         "gpt-4-32k": { "pricing": {"prompt_tokens": 0.06, "completion_tokens": 0.12}},
#         "gpt-3.5-turbo-1106": { "pricing": {"prompt_tokens": 0.001, "completion_tokens": 0.0020}},

#         # guesses on 2023.11.16
#         "gpt-3.5-turbo": { "pricing": {"prompt_tokens": 0.001, "completion_tokens": 0.0020}},
#         "gpt-3.5-turbo-0613": { "pricing": {"prompt_tokens": 0.001, "completion_tokens": 0.0020}},

#         "no-pricing": { "pricing": {"prompt_tokens": 0, "completion_tokens": 0}}
#     }


#     def __init__(self, api_key=os.environ["OPENAI_API_KEY"]):
#         super().__init__()  # Call the parent class's __init__ method
#         self.api_key = api_key
#         self.client = OpenAI(api_key=self.api_key)


#     def create_completion(self, llm_params, messages):

#         log("info", self, f"""START""")
#         log("info", self, f"""INPUT: llm_params: {llm_params}""")
#         log("info", self, f"""INPUT: messages: {json.dumps(messages,indent=4)}""")

#         llm_response = None
#         llm_generation_time = 0
#         try:
#             start_time = time.time()
#             llm_response = self.client.chat.completions.create(
#                 **llm_params,
#                 messages=messages
#             )
#             end_time = time.time()
#             llm_generation_time = end_time - start_time
#         except Exception as e:
#             log("error", self, f"""Error while trying to generate a completion: {e}""")
            

#         llm_response = llm_response.model_dump()
#         llm_response["usage"]["generation_time"] = llm_generation_time

#         log("info", self, f"""RETURNING: {json.dumps(llm_response,indent=4)}""")

#         return llm_response





# from datetime import datetime
# import os
# import fnmatch
# import ast
# import numpy as np

# from sqlalchemy import Column, Integer, String, DateTime, JSON, create_engine
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.dialects.postgresql import JSONB

# Base = declarative_base()

# class PromptEngineeringExperimentsData(Base):
#     __tablename__ = 'prompt_engineering_experiments_data'

#     id = Column(Integer, primary_key=True)  # Add this line
#     datetime_generated = Column(DateTime)
#     exp_name = Column(String(255))
#     exp_ver = Column(String(255))
#     status = Column(String(255))
#     generation = Column(String)
#     llm_params = Column(JSONB)
#     messages = Column(JSONB)
#     messages_template = Column(JSONB)
#     data = Column(JSONB)
#     llm_usage = Column(JSONB)

#     def __init__(self, db_config):
#         db_config_str = f"""postgresql+psycopg2://{db_config["POSTGRES_USER"]}:{db_config["POSTGRES_PASSWORD"]}@{db_config["POSTGRES_HOST"]}:5432/{db_config["POSTGRES_DB"]}"""

#         self.engine = create_engine(db_config_str)
#         Session = sessionmaker(bind=self.engine)
#         self.session = Session()




# class PromptEngineeringExperiment():

#     # "name": { "versions": { "v": { "message_templates": [] } } }
#     experiments = {}
    
#     def __init__(self, name, ver=None, message_templates=None, llm=None, llm_params=None, test_data=None, reports_folder="reports/", logs_folder="logs/", assessors=None, db_config=None):
#         self.name = name
#         self.ver = ver
#         self.message_templates = message_templates
#         self.test_data = test_data or []
#         self.llm_params = llm_params
#         self.reports_folder = reports_folder
#         self.assessors = assessors
#         self.logs_folder = logs_folder
#         self.llm = llm
#         self.db_config = db_config
#         if db_config:
#             self.db = PromptEngineeringExperimentsData(self.db_config)

#         self.experiments[self.name] = {
#             "versions": {
#                 self.ver: {
#                     "message_templates": self.message_templates
#                 }
#             },
#             "llm_params": [
#                 self.llm_params
#             ]
#         }

#         # self.aicht = AIChampTools(logs_folder=logs_folder)


#     def _obj2dict(self,obj):
#         if isinstance(obj, str):
#             return json.loads(obj)
#         elif not isinstance(obj, dict):
#             return obj.__dict__
#         else:
#             return obj



#     def save_results2(self, results, overwrite=False, ver=None):
#         results["exp_name"] = self.name
#         results["exp_ver"] = ver or self.ver

#         json_columns = ["data", "llm_params", "llm_usage"]
#         list_columns = ["messages", "messages_template"]
#         assessor_columns = [col for col in results.columns if 'assessor.' in col]

#         results["llm_usage"] = results["llm_usage"].apply(lambda x: x.__dict__ if isinstance(x, LLMUsage) else x)

#         for column in [*json_columns, *list_columns, *assessor_columns]:
#             results[column] = results[column].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

#         # Create an instance of PromptEngineeringExperimentsData
#         results.to_sql('prompt_engineering_experiments_data', self.db.engine, if_exists='replace' if overwrite else 'append', index=False)

#         return results


#     def save_results(self, results, overwrite=False, ver=None):
#         ver = ver or self.ver
#         filename = f"{self.reports_folder}experiment_{self.name}_v{ver}.csv"

#         json_columns = ["data", "llm_params", "llm_usage"]
#         list_columns = ["messages", "messages_template"]
#         assessor_columns = [col for col in results.columns if 'assessor.' in col]

#         results["llm_usage"] = results["llm_usage"].apply(lambda x: x.__dict__ if isinstance(x, LLMUsage) else x)
#         # print(results["llm_usage"])

#         for column in [*json_columns, *list_columns, *assessor_columns]:
#             results[column] = results[column].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)

#         # If file does not exist, or we need to overwrite the data 
#         if overwrite or not os.path.isfile(filename):
#             results.to_csv(filename, index=False)
            
#         else: 
#             # If file exists, read the existing data
#             df_existing = pd.read_csv(filename)
#             # Concatenate the existing data with the new data
#             results = pd.concat([df_existing, results], ignore_index=True)
#             # Write the combined data back to the file
#             results.to_csv(filename,index=False)
        
#         return results

    
#     def load_results(self, ver=None, flatten=False, llm_params=None, deserialize=True, flatten_exclude=[], from_db=False):
#         """
#         if no ver, we are loading from the current version
        
#         ver, llm_params - filters
#         """

#         ver = ver or self.ver

#         if from_db:
#             Session = sessionmaker(bind=self.db.engine)
#             session = Session()

#             log("info", self, f"""Trying to get data for {self.name}/{ver}""")

#             # Query the database
#             data = session.query(PromptEngineeringExperimentsData).filter(
#                 PromptEngineeringExperimentsData.exp_name == self.name,
#                 PromptEngineeringExperimentsData.exp_ver == ver
#             ).all()

#             # Close the Session
#             session.close()
        
#             data_df = pd.DataFrame([d.__dict__ for d in data])


#         # from file
#         else:
#             report_filepath = f"{self.reports_folder}experiment_{self.name}_v{ver}.csv"
#             # Load the CSV file into a DataFrame
#             data_df = pd.read_csv(report_filepath)

#         json_columns = ["data", "llm_params", "llm_usage"]
#         list_columns = ["messages", "messages_template"]
#         assessor_columns = [col for col in data_df.columns if 'assessor.' in col]

#         if deserialize:
#             for column in [*json_columns, *list_columns, *assessor_columns]:
#                 data_df[column] = data_df[column].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

#         if flatten:
#             for column in [*json_columns, *assessor_columns]:
#                 expanded = pd.json_normalize(data_df[column])
#                 expanded.columns = f"{column}." + expanded.columns
#                 data_df = data_df.drop(columns=[column]).join(expanded)

#         return data_df


#         # # test_results = self.aicht.load_synth_data(report_filepath)#, flatten=True
#         # test_results.replace({np.nan: None}, inplace=True)

#         # test_results["data"] = test_results["data"].apply(json.loads)
#         # test_results["llm_params"] = test_results["llm_params"].apply(json.loads)
#         # test_results["llm_usage"] = test_results["llm_usage"].apply(json.loads)
#         # test_results["messages"] = test_results["messages"].apply(json.loads)
#         # test_results["messages_template"] = test_results["messages_template"].apply(json.loads)
#         # assessor_columns = []

        
#         # for c in test_results.columns:
#         #     if "assessor." in c:
#         #         test_results[c] = test_results[c].apply(lambda x: json.loads(x) if pd.notnull(x) else x)
#         #         # # convert second level data into strings
#         #         # test_results[c] = test_results[c].apply(lambda x: {k: json.dumps(v) if isinstance(v, dict) else v for k, v in x.items()} if pd.notnull(x) else x)
#         #         assessor_columns.append(c)

#         # if flatten:
#         #     for column in ["data", "llm_params", "llm_usage", *assessor_columns]:
#         #         if column in flatten_exclude:
#         #             continue
#         #         if isinstance(test_results[column][0],str):
#         #             expanded = pd.json_normalize(test_results[column].apply(json.loads))
#         #         else:
#         #             expanded = pd.json_normalize(test_results[column])
#         #         expanded.columns = f"{column}." + expanded.columns
#         #         test_results = test_results.drop(columns=[column]).join(expanded)        
#         # # test_results["messages"] = test_results["messages"].apply(json.loads)

#         # return test_results    
    
    
#     def run(self, test_data, reps=1, ver=None, max_data_points=0, llm_params=None, assessors=None):
#         sbs_logger.info("START")
#         sbs_logger.info(f"llm_params: {llm_params}")

#         llm_params = copy.deepcopy(llm_params) or copy.deepcopy(self.llm_params)
#         llm_params = { **llm_params, "n": reps }
#         # llm_params["timeout"] = llm_params.get("timeout") or 30
#         sbs_logger.info(f"llm_params after modif: {llm_params}")


#         message_templates = self.message_templates or [
#             {"role": "system", "content": "Act as a fruit professional. Always respond in json format with the keys: fruit, color"},
#             {"role": "user", "content": "Which color is a {fruit}?"}
#         ]
#         sbs_logger.info(f"message_templates: {message_templates}")


#         sbs_logger.info(f"test_data: {test_data}")


#         synth_data = []

#         data_points_processed = 0

#         if test_data:
#             # Generate synthetic data for each data point
#             for i, dp in enumerate(test_data):

#                 sbs_logger.info(f"DATAPOINT {i}: {dp}")


#                 if max_data_points and data_points_processed>=max_data_points:
#                     sbs_logger.info(f"Maximum number of data points to process reached.")
#                     break

#                 # Populate a copy of messages with the data point values
#                 messages_populated = copy.deepcopy(message_templates)
#                 for msg in messages_populated:
#                     msg["content"] = msg["content"].format(**dp)

#                 llm_response = {}


#                 sbs_logger.info(f"messages passed to AI: {json.dumps(messages_populated, indent=4)}")


#                 llm_response = self.llm.create_completion(llm_params=llm_params, messages=messages_populated)


#                 sbs_logger.info(f'How many choices? {llm_response["choices"]}')


#                 for j, choice in enumerate(llm_response["choices"]):

#                     sbs_logger.info(f"Data point {i+1}/{len(test_data)} (limit: {max_data_points}). Repetition {j+1}/{reps}")

#                     res = choice["message"]["content"]

#                     prompt_tokens = llm_response['usage']['prompt_tokens'] or 0
#                     completion_tokens = llm_response['usage']['completion_tokens']/reps or 0
#                     llm_generation_time = llm_response['usage']['generation_time']/reps or 0
#                     total_tokens = prompt_tokens+completion_tokens
#                     llm_usage = LLMUsage(
#                         prompt_tokens=prompt_tokens,
#                         completion_tokens=completion_tokens,
#                         total_tokens=total_tokens,
#                         generation_time=llm_generation_time or 0
#                     )
#                     llm_usage.total_cost = self.llm.execution_cost(model=llm_params['model'],llm_usage=llm_usage)

#                     synth_data.append({
#                         "datetime_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                         "status": "",
#                         "generation": res or "",
#                         "llm_params": llm_params,
#                         "messages": messages_populated,
#                         "messages_template": message_templates,
#                         "data": dp or "",
#                         "llm_usage": llm_usage
#                     })
#         else:
#             messages_populated = message_templates

#             llm_response = self.llm.create_completion(llm_params=llm_params, messages=messages_populated)

#             for j, choice in enumerate(llm_response["choices"]):

#                 sbs_logger.info(f"Repetition {j+1}/{reps}")

#                 res = choice["message"]["content"]

#                 prompt_tokens = llm_response['usage']['prompt_tokens'] or 0
#                 completion_tokens = llm_response['usage']['completion_tokens'] or 0
#                 llm_generation_time = llm_response['usage']['generation_time']/reps or 0
#                 total_tokens = prompt_tokens+completion_tokens

#                 llm_usage = LLMUsage(
#                     prompt_tokens=prompt_tokens,
#                     completion_tokens=completion_tokens,
#                     total_tokens=total_tokens,
#                     generation_time=llm_generation_time or 0
#                 )
#                 llm_usage.total_cost = self.llm.execution_cost(model=llm_params['model'],llm_usage=llm_usage)

#                 synth_data.append({
#                     "datetime_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     "status": "",
#                     "generation": res or "",
#                     "llm_params": llm_params,
#                     "messages": messages_populated,
#                     "messages_template": message_templates,
#                     "data": "",
#                     "llm_usage": llm_usage
#                 })


#         sbs_logger.info(f"synth_data generated: {synth_data}")


#         synth_data_df = pd.DataFrame(synth_data)


#         sbs_logger.info(f"assessors: {assessors}")


#         if assessors:
#             for assr in assessors:
#                 sbs_logger.info(f"assessing using {assr}")

#                 assr["function"](synth_data_df, *assr["input"])

#         log("info", self, f"""1 Saving {len(synth_data_df)} results""")

#         results = self.save_results2(synth_data_df)

#         log("info", self, f"""2 Saving {len(results)} results""")

#         log("info", self, f"""Trying to get data for {self.name}/{ver}""")


#         return results
        
        

#     def init_ver(self,ver,message_templates,llm_params=None,assessors=None):
#         # print(f"trying to add {ver} to self.experiments: {self.experiments}")
        
#         # if ver in self.experiments[self.name]["versions"]:
#         #     # print(f"Experiment version exists, no need to init.")
#         #     return
        
#         self.experiments[self.name] = {
#             "versions": {
#                 ver: {
#                     "message_templates": message_templates
#                 },
#                 "llm_params": []
#             }
#         }

#         self.ver = ver
#         self.message_templates = message_templates
#         self.llm_params = llm_params or self.llm_params
#         self.assessors = assessors or self.assessors


#     def assess(self,assessor,ver=None,in_chunks=0,max_entries=0):

#         ver = ver or self.ver

#         sbs_logger.info('START')
#         sbs_logger.info(f'ver: {ver}')
#         sbs_logger.info(f'assessor: {assessor}')
#         sbs_logger.info(f'max_entries: {max_entries}')

#         synth_data_df = self.load_results(ver=ver)
#         sbs_logger.info(f'number of results to assess: {len(synth_data_df)}')


#         if not in_chunks:
#             in_chunks = len(synth_data_df)
#         groups = synth_data_df.groupby(np.arange(len(synth_data_df)) // in_chunks)
#         sbs_logger.info(f'type(groups): {type(groups)}')


#         for i, (group_name, chunk) in enumerate(groups):
#             sbs_logger.info(f'Assessing group {i+1}/{len(groups)}')

#             # Apply the function to the chunk
#             assessor["function"](chunk, *assessor["input"])

#             # Add missing columns to synth_data_df
#             missing_cols = set(chunk.columns) - set(synth_data_df.columns)
#             for col in missing_cols:
#                 synth_data_df[col] = np.nan

#             # Replace the corresponding chunk in synth_data_df
#             synth_data_df.iloc[i*in_chunks:(i+1)*in_chunks] = chunk
#             self.save_results(results=synth_data_df, overwrite=True, ver=ver)
  


    
#     # deprecated
#     def deprecated_assess(self,assessor,ver=None,max_entries=0):

#         ver = ver or self.ver
#         test_results = self.load_results(ver=ver)
#         report_filepath = f"{self.reports_folder}experiment_{self.name}_v{ver}.csv"


#         assessor_column_name = f"assessor.{assessor['function'].__name__}"

#         test_results_dict = test_results.to_dict('records')
#         assessor_input_0l_values = []
#         for kl in assessor["input_0level"]:
#             value = reduce(operator.getitem, kl, test_results_dict)
#             assessor_input_0l_values.append(value)
#             print(f"input_0level: {kl} -> {value}")

#         j = 0
#         for i,row in test_results.iterrows():
#             if max_entries and j>= max_entries:
#                 break
#             if not assessor_column_name in test_results.columns:
#                 test_results[assessor_column_name] = None
#             # print(f"anything here? '{test_results.at[i, assessor_column_name]}'")            
#             if test_results.at[i, assessor_column_name]:
#                 # print(f"skipping")
#                 continue
            
#             assessor_input_values = []
#             for kl in assessor["input"]:
#                 # print(f"kl: {kl}")
#                 value = reduce(operator.getitem, kl, row)
#                 # print(f"value ({type(value)}): {value}")
#                 assessor_input_values.append(value)
            
#             assessment = assessor["function"](*assessor_input_values, *assessor_input_0l_values, *assessor["input2"])
#             # print(f"{i} ({type(assessment)}): {assessment}")

#             # Add the assessment to the column
#             test_results.at[i, assessor_column_name] = assessment

#             j += 1
        
#         self.save_results(results=test_results, overwrite=True, ver=ver)
        



# import json
# from openai import OpenAI, ChatCompletion, APITimeoutError
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


# class PromptEngineeringExperimentAssessors():


#     def __init__(self):
#         pass


#     def assess_json_valid(self, results):

#         func_name = inspect.currentframe().f_code.co_name
#         results[f'assessor.{func_name}'] = None

#         for index, row in results.iterrows():

#             json_valid = False
#             try:
#                 row["generation"]=json.loads(row["generation"].replace("\n",""))
#                 json_valid = True
#             except Exception as e:
#                 pass
#                 # print(f'Not a valid JSON: {row["generation"]}')

#             results.loc[index, f'assessor.{func_name}'] = json.dumps({ "valid": json_valid })



#     def assess_len(self, results):

#         func_name = inspect.currentframe().f_code.co_name
#         results[f'assessor.{func_name}'] = results['generation'].apply( lambda x: json.dumps({"len":len(x)}) )



#     def assess_str_diff_from_most_common(self, results):

#         func_name = inspect.currentframe().f_code.co_name

#         variability_matrix = []

#         strings = list(results["generation"].unique())

#         for i, v in enumerate(strings):
#             variability_matrix.append([])#rowi["generation"]

#             for j, v2 in enumerate(strings):
#                 vectorizer = TfidfVectorizer().fit_transform([v, v2])
#                 vectors = vectorizer.toarray()
#                 csim = cosine_similarity(vectors)
#                 variability_matrix[i].append(csim[0][1])

#         variability_matrix_df = pd.DataFrame(variability_matrix,columns=strings)
#         variability_matrix_df.index = strings

#         highest_avg_row = variability_matrix_df.loc[[variability_matrix_df.mean(axis=1).idxmax()]]
#         highest_avg_row = highest_avg_row.reset_index(drop=True)

#         highest_avg_row = highest_avg_row.iloc[0].to_dict()

#         results[f'assessor.{func_name}'] = results['generation'].map(highest_avg_row).apply(lambda x: json.dumps({"result": x}))

#         return results
    


#     def assess_response(self, results, llm, llm_params):

#         func_name = inspect.currentframe().f_code.co_name

#         if f'assessor.{func_name}' not in results.columns:
#             results[f'assessor.{func_name}'] = None

#         mask = results[f'assessor.{func_name}'].isnull()

#         system_message = """
#             Here's a conversation between an Assistant and a User:
#             --------
#             {messages}
#             --------
            
#             Your objective is to assess if the Assistant is providing the information the User is requesting.

#             The only possible result statuses:
#             - providing
#             - ignoring
#             - rejecting
            
#             You must reply in JSON format with the fields:
#             - "reasoning": your step by step detailed reasoning of the assessment
#             - "result": result status
#         """

#         # results['tmp.{func_name}.message_history'] = results.apply(lambda row: row["messages"] + [{"role": "assistant", "content": row["generation"]}], axis=1)
#         # results["tmp.{func_name}.system_message"] = results['tmp.{func_name}.message_history'].apply(lambda x: [ { "role": "system", "content": system_message.format(messages=str(x)) }])
#         # results[f'assessor.{func_name}'] = results["tmp.{func_name}.system_message"].apply(lambda x: llm.create_completion(messages=x, llm_params=llm_params)["choices"][0]["message"]["content"])

#         # results[f'assessor.{func_name}'] = results[f'assessor.{func_name}'].apply(lambda x: x.replace('```json', '', 1) if x.startswith('```json') else x)
#         # results[f'assessor.{func_name}'] = results[f'assessor.{func_name}'].apply(lambda x: x.replace('```', '', 1) if x.endswith('```') else x)


#         results.loc[mask, 'tmp.{func_name}.message_history'] = results.loc[mask].apply(lambda row: row["messages"] + [{"role": "assistant", "content": row["generation"]}], axis=1)
#         results.loc[mask, "tmp.{func_name}.system_message"] = results.loc[mask, 'tmp.{func_name}.message_history'].apply(lambda x: [ { "role": "system", "content": system_message.format(messages=str(x)) }])
#         results.loc[mask, f'assessor.{func_name}'] = results.loc[mask, "tmp.{func_name}.system_message"].apply(lambda x: llm.create_completion(messages=x, llm_params=llm_params)["choices"][0]["message"]["content"])

#         results.loc[mask, f'assessor.{func_name}'] = results.loc[mask, f'assessor.{func_name}'].apply(lambda x: x.replace('```json', '', 1) if x.startswith('```json') else x.rstrip('```'))
#         results.loc[mask, f'assessor.{func_name}'] = results.loc[mask, f'assessor.{func_name}'].apply(lambda x: x.replace('```', '', 1) if x.endswith('```') else x)

#         tmp_columns = results.filter(like='tmp.').columns
#         results.drop(columns=tmp_columns, inplace=True)
