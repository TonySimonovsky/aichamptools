import copy
import json
import time
import os

from ..core import LLM
from ..utilities import log_message

from openai import OpenAI, ChatCompletion, APITimeoutError
import tiktoken




class vLLM(LLM):

    vendor=""
    api_hoster=""

    def __init__(self, api_key, api_url=None, vendor=None, api_hoster=None, log_on=True):

        super().__init__(log_on=log_on)

        self.log_on = log_on if log_on is not None else self.log_on

        self.api_url = api_url
        self.api_key = api_key
        self.vendor = vendor
        self.api_hoster = api_hoster

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )


    def create_completion(self, llm_params, messages, output_v=0.02):
        """

        output_v - different output versions:
        0.01 - original output
        0.02 - unified+original output
        
        
        """

        log_message(self.logger, "info", self, f"""START""")
        log_message(self.logger, "info", self, f"""INPUT: llm_params: {llm_params}""")
        log_message(self.logger, "info", self, f"""INPUT: messages: {json.dumps(messages,indent=4)}""")

        llm_response = None
        llm_generation_time = 0
        try:
            start_time = time.time()
            llm_response = self.client.chat.completions.create(
                **llm_params,
                messages=messages
            )
            end_time = time.time()
            llm_generation_time = end_time - start_time
        except Exception as e:
            log_message(self.logger, "error", self, f"""Error while trying to generate a completion: {e}""")
            

        llm_response = llm_response.model_dump()
        llm_response["usage"]["generation_time"] = llm_generation_time
        if output_v==0.02:
            llm_response_original = copy.deepcopy(llm_response)
            llm_response = {}
            llm_response["original"] = llm_response_original
            llm_response["unified"] = {
                "message": llm_response_original["choices"][0]["message"]["content"],
                "role": llm_response_original["choices"][0]["message"]["role"],
                "llm_params": llm_params,
                "usage": llm_response_original["usage"]
            }

        log_message(self.logger, "info", self, f"""RETURNING: {json.dumps(llm_response,indent=4)}""")

        return llm_response



