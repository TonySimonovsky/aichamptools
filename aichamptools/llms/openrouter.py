import copy
import json
import time

from ..core import LLM
from ..utilities import log_message

import requests


class LLMsOpenRouter(LLM):

    vendor=""
    api_hoster="OpenRouter"

    models = {
        # from https://openrouter.ai/docs#models on 2023.12.29
        "anthropic/claude-2": { "pricing": {"prompt_tokens": 0.008, "completion_tokens": 0.024}, "context_window": 200000 },
        "anthropic/claude-2.0": { "pricing": {"prompt_tokens": 0.008, "completion_tokens": 0.024}, "context_window": 100000 },
        "anthropic/claude-instant-v1": { "pricing": {"prompt_tokens": 0.00163, "completion_tokens": 0.00551}, "context_window": 100000 },
        "meta-llama/llama-3-70b-instruct": { "pricing": {"prompt_tokens": 0.0008, "completion_tokens": 0.0008}, "context_window": 8192 },

        "no-pricing": { "pricing": {"prompt_tokens": 0, "completion_tokens": 0}}
    }



    def __init__(self, api_key, api_url=None, log_on=True):

        super().__init__(log_on=log_on)

        self.log_on = log_on if log_on is not None else self.log_on

        self.api_key = api_key
        self.api_url = api_url

        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'HTTP-Referer': 'https://localhost',
            'Content-Type': 'application/json'
        }




    def create_completion(self, llm_params, messages, output_v=0.02):

        log_message(self.logger, "info", self, f"""START""")
        log_message(self.logger, "info", self, f"""INPUT: llm_params: {llm_params}""")
        log_message(self.logger, "info", self, f"""INPUT: messages: {json.dumps(messages,indent=4)}""")

        llm_response = None
        llm_generation_time = 0
        try:
            start_time = time.time()

            llm_response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers=self.headers, data=json.dumps({**llm_params,"messages":messages}))

            end_time = time.time()
            llm_generation_time = end_time - start_time

            log_message(self.logger, "error", self, f"""LLM response received: {llm_response.json()}""")


        except Exception as e:
            log_message(self.logger, "error", self, f"""Error while trying to generate a completion: {e}""")
            


        llm_response = llm_response.json()
        if "usage" in llm_response:
            llm_response["usage"]["generation_time"] = llm_generation_time
        else:
            llm_response["usage"] = { "generation_time": llm_generation_time }


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





        log_message(self.logger, "info", self, f"""RETURNING: {json.dumps(llm_response,indent=4)}""")

        return llm_response

