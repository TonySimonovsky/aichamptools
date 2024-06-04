import copy
import json
import time

from ..core import LLM
from ..utilities import log_message

import requests


class LLMsOnPerplexity(LLM):

    vendor=""
    api_hoster="Perplexity"

    models = {
    }



    def __init__(self, api_key, api_url="https://api.perplexity.ai/chat/completions", vendor=None, api_hoster=None, log_on=True):

        super().__init__(log_on=log_on)

        self.log_on = log_on if log_on is not None else self.log_on

        self.api_url = api_url
        self.api_key = api_key


        self.headers = {
            "accept" : "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.api_key}"
        }




    def create_completion(self, llm_params, messages, output_v=0.02):

        log_message(self.logger, "info", self, f"""START""")
        log_message(self.logger, "info", self, f"""INPUT: llm_params: {llm_params}""")
        log_message(self.logger, "info", self, f"""INPUT: messages: {json.dumps(messages,indent=4)}""")

        llm_response = None
        llm_generation_time = 0
        try:
            start_time = time.time()


            llm_response = requests.post(self.api_url, headers=self.headers, json={ **llm_params, "messages": messages })

            # llm_response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=self.headers, data=json.dumps({**llm_params,"messages":messages}))

            # print(type(llm_response.json()))

            end_time = time.time()
            llm_generation_time = end_time - start_time

            log_message(self.logger, "info", self, f"""LLM response received: {llm_response.json()}""")


        except Exception as e:
            log_message(self.logger, "error", self, f"""Error while trying to generate a completion: {e}""")
            # print(f"""Error while trying to generate a completion: {e}""")
        

        # print(f"\n\n\nllm_response.json() = {llm_response.json()}\n\n\n")
        llm_response = llm_response.json()
        if "usage" in llm_response:
            llm_response["usage"]["generation_time"] = llm_generation_time
        else:
            llm_response["usage"] = { "generation_time": llm_generation_time }


        if output_v==0.02:
            llm_response_original = copy.deepcopy(llm_response)
            llm_response = {}
            llm_response["original"] = llm_response_original
            generated_message = llm_response_original["choices"][0]["message"]
            llm_response["unified"] = {
                "message": generated_message["content"],
                "role": generated_message["role"],
                "llm_params": llm_params,
                "usage": llm_response_original["usage"]
            }

        log_message(self.logger, "info", self, f"""RETURNING: {json.dumps(llm_response,indent=4)}""")

        return llm_response


