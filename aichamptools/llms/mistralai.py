import copy
import json
import time

from ..core import LLM
from ..utilities import log_message

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


class LLMMistral(LLM):

    vendor="MistralAI"
    api_hoster="MistralAI"


    models = {
        # from https://docs.mistral.ai/platform/pricing/ on 2023.12.15
        # "mistral-tiny": { "pricing": {"prompt_tokens": 0.14*1.1/1000, "completion_tokens": 0.42*1.1/1000}, "context_window": 32000 },
        "mistral-small": { "pricing": {"prompt_tokens": 2/1000, "completion_tokens": 6/1000}, "context_window": 32000 },
        "mistral-medium": { "pricing": {"prompt_tokens": 2.7/1000, "completion_tokens": 8.1/1000}, "context_window": 32000 },
        "mistral-large-latest": { "pricing": {"prompt_tokens": 8/1000, "completion_tokens": 20/1000}, "context_window": 32000 },
        "no-pricing": { "pricing": {"prompt_tokens": 0, "completion_tokens": 0}}
    }

    def __init__(self, api_key, api_url=None, vendor=None, api_hoster=None, log_on=True):

        super().__init__(log_on=log_on)

        self.log_on = log_on if log_on is not None else self.log_on

        self.api_key = api_key
        self.api_url = api_url
        self.client = MistralClient(api_key=self.api_key)
        self.requires_user_message = True
    

    def create_completion(self, llm_params, messages, output_v=0.02):

        llm_params_copy = copy.deepcopy(llm_params)

        log_message(self.logger, "info", self, f"""START""")
        log_message(self.logger, "info", self, f"""INPUT: llm_params: {llm_params_copy}""")
        log_message(self.logger, "info", self, f"""INPUT: messages: {json.dumps(messages,indent=4)}""")
        log_message(self.logger, "info", self, f"""INPUT: api_key passed: {bool(self.api_key)}""")

        # # checking of there the last message is from the user (Mistral LLM requirement)
        # if messages[-1]["role"] != "user":
        #     messages.append({"role": "user", "content":"[... ignore this message and continue following your instructions]"})

        try:
            messages = [
                ChatMessage(role=m["role"], content=m["content"]) for m in messages
            ]
        except Exception as e:
            log_message(self.logger, "error", self, f"""Couldn't convert to ChatMessage objects: {e}""")


        reps = llm_params_copy.pop("n", None) or 1

        llm_responses_all = { "choices": [], "usage": { "prompt_tokens": 0, "completion_tokens": 0, "generation_time": 0 } }
        for i in range(reps):

            log_message(self.logger, "info", self, f"""rep {i+1}/{reps}...""")

            llm_response = None
            llm_generation_time = 0

            log_message(self.logger, "info", self, f"""ACTUALLY BEING SENT TO THE MODEL:""")
            log_message(self.logger, "info", self, f"""llm_params: {llm_params_copy}""")
            log_message(self.logger, "info", self, f"""messages: {messages}""")


            try:
                start_time = time.time()

                llm_response = self.client.chat(
                    **llm_params_copy,
                    messages=messages
                )
                end_time = time.time()
                llm_generation_time = end_time - start_time
                log_message(self.logger, "info", self, f"""llm_response {i+1}/{reps} before model_dump: {llm_response}""")

            except Exception as e:
                log_message(self.logger, "error", self, f"""Error while trying to generate a completion: {e}""")

            llm_response = llm_response.model_dump()

            log_message(self.logger, "info", self, f"""llm_response {i+1}/{reps}: {llm_response}""")

            llm_response["usage"]["generation_time"] = llm_generation_time

            llm_responses_all["choices"].append(llm_response["choices"][0])

            llm_responses_all["usage"]["prompt_tokens"] = llm_responses_all["usage"].get("prompt_tokens") or llm_response["usage"]["prompt_tokens"]
            llm_responses_all["usage"]["completion_tokens"] += llm_response["usage"]["completion_tokens"]
            llm_responses_all["usage"]["generation_time"] += llm_generation_time


        if output_v==0.02:
            llm_response_original = copy.deepcopy(llm_responses_all)
            llm_responses_all = {}
            llm_responses_all["original"] = llm_response_original
            llm_responses_all["unified"] = {
                "message": llm_response_original["choices"][0]["message"]["content"],
                "role": llm_response_original["choices"][0]["message"]["role"],
                "llm_params": llm_params,
                "usage": llm_response_original["usage"]
            }


        log_message(self.logger, "info", self, f"""RETURNING: {json.dumps(llm_responses_all,indent=4,default=str)}""")

        return llm_responses_all

