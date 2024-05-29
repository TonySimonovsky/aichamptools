import copy
import json
import time

from ..core import LLM
from ..utilities import log_message

import anthropic


class LLMAnthropic(LLM):
    vendor="Anthropic"
    api_hoster="Anthropic"

    models = {
        # https://docs.anthropic.com/claude/docs/models-overview
        "claude-3-opus-20240229": { "pricing": {"prompt_tokens": 0.015, "completion_tokens": 0.075}, "context_window": 200000, "max_output": 4096 },
        "claude-3-sonnet-20240229": { "pricing": {"prompt_tokens": 0.003, "completion_tokens": 0.015}, "context_window": 200000, "max_output": 4096 },
        "claude-3-haiku-20240307": { "pricing": {"prompt_tokens": 0.00025, "completion_tokens": 0.00125}, "context_window": 200000, "max_output": 4096 },
        "claude-2.1": { "pricing": {"prompt_tokens": 0.008, "completion_tokens": 0.024}, "context_window": 200000, "max_output": 4096 },
        "claude-2.0": { "pricing": {"prompt_tokens": 0.008, "completion_tokens": 0.024}, "context_window": 100000, "max_output": 4096 },
        "claude-instant-1.2": { "pricing": {"prompt_tokens": 0.008, "completion_tokens": 0.024}, "context_window": 100000, "max_output": 4096 },

        "no-pricing": { "pricing": {"prompt_tokens": 0, "completion_tokens": 0}}
    }

    api_key = None



    def __init__(self, api_key=None, api_url=None, vendor=None, api_hoster=None, log_on=True):

        super().__init__(log_on=log_on)

        # if not api_key:
        self.api_key = api_key

        self.log_on = log_on if log_on is not None else self.log_on

        self.client = anthropic.Anthropic(api_key=api_key)




    def create_completion(self, llm_params, messages, output_v=0.02):
        log_message(self.logger, "info", self, f"""START""")
        log_message(self.logger, "info", self, f"""INPUT: llm_params: {llm_params}""")
        log_message(self.logger, "info", self, f"""INPUT: messages: {json.dumps(messages,indent=4)}""")

        system_prompt = None
        # Filter out the system message and update the messages list
        filtered_messages = []
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            else:
                filtered_messages.append(message)

        # Use filtered_messages from this point forward
        messages = filtered_messages

        llm_response = None
        llm_generation_time = 0
        try:
            start_time = time.time()

            # Ensure to use the updated messages list without the system message
            if system_prompt:
                llm_response = self.client.messages.create(
                    system=system_prompt,
                    **llm_params,
                    messages=messages,
                )
            else:
                llm_response = self.client.messages.create(
                    **llm_params,
                    messages=messages,
                )
            end_time = time.time()
            llm_generation_time = end_time - start_time
        except Exception as e:
            log_message(self.logger, "error", self, f"""Error while trying to generate a completion: {e}""")

        llm_response = llm_response.model_dump()
        llm_response["usage"]["generation_time"] = llm_generation_time


        if output_v==0.02:
            llm_response_original = copy.deepcopy(llm_response)

            log_message(self.logger, "info", self, f"""OUTPUT: llm_response["usage"]: {llm_response["usage"]}""")
            log_message(self.logger, "info", self, f"""OUTPUT: llm_response_original["usage"]: {llm_response_original["usage"]}""")

            llm_response = {}
            llm_response["original"] = llm_response_original
            llm_response["unified"] = {
                "message": llm_response_original["content"][0]["text"],
                "role": llm_response_original["role"],
                "llm_params": llm_params,
                "usage": llm_response_original["usage"],
                "cost": self.execution_cost(model=llm_params["model"], llm_usage=llm_response_original["usage"])
            }

        log_message(self.logger, "info", self, f"""RETURNING: {json.dumps(llm_response,indent=4)}""")

        return llm_response

