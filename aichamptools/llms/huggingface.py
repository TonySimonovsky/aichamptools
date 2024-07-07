import copy
import json
import time

from ..core import LLM
from ..utilities import log_message

import requests


class LLMsOnHF(LLM):

    vendor=""
    api_hoster="HuggingFace"

    models = {
    }



    def __init__(self, api_key, api_url=None, vendor=None, api_hoster=None, log_on=True):

        super().__init__(log_on=log_on)

        self.log_on = log_on if log_on is not None else self.log_on

        self.api_url = api_url
        self.api_key = api_key


        self.headers = {
            "Accept" : "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json" 
        }




    def create_completion(self, llm_params, messages, output_v=0.02):

        log_message(self.logger, "info", self, f"""START""")
        log_message(self.logger, "info", self, f"""INPUT: llm_params: {llm_params}""")
        log_message(self.logger, "info", self, f"""INPUT: messages: {json.dumps(messages,indent=4)}""")

        llm_response = None
        llm_generation_time = 0
        try:
            start_time = time.time()


            llm_response = requests.post(self.api_url, headers=self.headers, json={"inputs": self.messages_to_chatml(messages=messages),"parameters": {}})

            # llm_response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=self.headers, data=json.dumps({**llm_params,"messages":messages}))

            # print(type(llm_response.json()))

            end_time = time.time()
            llm_generation_time = end_time - start_time

            log_message(self.logger, "info", self, f"""LLM response received: {llm_response.json()}""")


        except Exception as e:
            log_message(self.logger, "error", self, f"""Error while trying to generate a completion: {e}""")
            # print(f"""Error while trying to generate a completion: {e}""")
        

        # print(f"\n\n\nllm_response.json() = {llm_response.json()}\n\n\n")
        llm_response = llm_response.json()[0]
        if "usage" in llm_response:
            llm_response["usage"]["generation_time"] = llm_generation_time
        else:
            llm_response["usage"] = { "generation_time": llm_generation_time }


        if output_v==0.02:
            llm_response_original = copy.deepcopy(llm_response)
            llm_response = {}
            llm_response["original"] = llm_response_original
            generated_message = self.chatml_to_messages(llm_response_original["generated_text"])[-1]
            llm_response["unified"] = {
                "message": generated_message["content"],
                "role": generated_message["role"],
                "llm_params": llm_params,
                "usage": llm_response_original["usage"]
            }

        log_message(self.logger, "info", self, f"""RETURNING: {json.dumps(llm_response,indent=4)}""")

        return llm_response





        log_message(self.logger, "info", self, f"""RETURNING: {json.dumps(llm_response,indent=4)}""")

        return llm_response

