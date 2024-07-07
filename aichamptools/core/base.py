import os
from ..utilities import setup_logging, log_message, enable_logging
import tiktoken
import re



class AIChampTools():

    def __init__(self, logs_folder="logs_aichamptools/",log_on=True):
        self.__version__ = '0.0.24'
        self.logs_folder = logs_folder
        # self.log_on = log_on


        # Setup logger
        enable_logging(enable=log_on)

        self.logger = setup_logging(logs_folder=logs_folder)




class LLMUsage(AIChampTools):

    def __init__(self, expected_prompt_tokens:int=0, prompt_tokens:int=0, expected_completion_tokens:int=0, completion_tokens:int=0, expected_total_tokens:int=0, total_tokens:int=0, expected_total_cost:float=0.0, total_cost:float=0.0, generation_time:float=0.0):

        super().__init__()

        # Expected number of prompt tokens
        self.expected_prompt_tokens = expected_prompt_tokens
        # Prompt tokens number received from LLM
        self.prompt_tokens = prompt_tokens
        self.expected_completion_tokens = expected_completion_tokens
        self.completion_tokens = completion_tokens
        self.expected_total_tokens = expected_total_tokens
        self.total_tokens = total_tokens
        self.expected_total_cost = expected_total_cost
        self.total_cost = total_cost
        self.generation_time = generation_time


    def __add__(self, other):
        if isinstance(other, LLMUsage):
            return LLMUsage(
                self.expected_prompt_tokens + other.expected_prompt_tokens,
                self.prompt_tokens + other.prompt_tokens,
                self.expected_completion_tokens + other.expected_completion_tokens,
                self.completion_tokens + other.completion_tokens,
                self.expected_total_tokens + other.expected_total_tokens,
                self.total_tokens + other.total_tokens,
                self.expected_total_cost + other.expected_total_cost,
                self.total_cost + other.total_cost,
                self.generation_time + other.generation_time
            )
        else:
            raise TypeError("Unsupported operand type. Both operands should be instances of LLMUsage.")


    @property
    def __dict__(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "generation_time": self.generation_time
        }





class LLM(AIChampTools):

    vendor = None
    models = {}

    def __init__(self, log_on=True):

        super().__init__(log_on=log_on)

        # a flag if the LLM requires a user message to get completion (for example, in OpenAI you can send only system message, but Mistral will throw an error if you do so with a user's message)
        self.requires_user_message = False
        self.log_on = log_on


    def execution_cost(self, model:str, llm_usage) -> float:

        log_message(self.logger, "info", self, f"""START""")
        log_message(self.logger, "info", self, f"""INPUT: model: {model}""")
        log_message(self.logger, "info", self, f"""INPUT: llm_usage: {llm_usage}""")

        if model in self.models:
            pricing = self.models[model]["pricing"]
            if isinstance(llm_usage, LLMUsage):
                prompt_tokens = llm_usage.prompt_tokens
                completion_tokens = llm_usage.completion_tokens
            elif isinstance(llm_usage, dict):
                prompt_tokens = llm_usage.get('input_tokens', 0)
                completion_tokens = llm_usage.get('output_tokens', 0)
            else:
                raise ValueError("llm_usage must be either an LLMUsage instance or a dict")

            log_message(self.logger, "info", self, f"""INPUT: prompt_tokens: {prompt_tokens}""")
            log_message(self.logger, "info", self, f"""INPUT: completion_tokens: {completion_tokens}""")

            cost = (prompt_tokens / 1000) * pricing["input_tokens"] + (completion_tokens / 1000) * pricing['output_tokens']
    
            log_message(self.logger, "info", self, f"""OUTPUT: cost: {cost:.10f}""")
        else:
            cost = "N/A"
            log_message(self.logger, "info", self, f"""OUTPUT: cost: {cost}""")


        return cost


    def num_tokens(self, text, encoding_name):

        try:
            encoding = tiktoken.encoding_for_model(encoding_name)
            num_tokens = len(encoding.encode(text))
        except:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-1106")
            num_tokens = len(encoding.encode(text))

        return num_tokens
    

    def messages_to_chatml(self, messages):
        prompt = ""


        for i, m in enumerate(messages):
            if i!=0:
                prompt += "\n"
            prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>"
        
        prompt += f"\n<|im_start|>assistant"

        return prompt


    def chatml_to_messages(self, prompt):
        
        # Ensure the prompt ends with a specific string if it doesn't already
        message_end_token = "<|im_end|>"
        if not prompt.rstrip().endswith(message_end_token):
            prompt = prompt.rstrip() + message_end_token

        # Pattern to match each message block within the prompt
        pattern = r"<\|im_start\|>(.*?)\n(.*?)<\|im_end\|>"
        
        # Find all matches of the pattern in the prompt
        matches = re.findall(pattern, prompt, re.DOTALL)
        
        # Convert each match into a dictionary and collect them into a list
        messages = [{'role': match[0], 'content': match[1]} for match in matches]
        
        return messages




    def messages_populate(self, messages, values_to_populate, inplace=False):
        """

        Populate messages with variables in them with the values.

        messages - list of dicts, same as what is accepted as input by LLMs (example: [{"role":"system","content":"some instructions"}])
        values_to_populate - dict of values to populate variables inside "content" of messages with
        inplace - by default, the method will create a copy of the messages, rather than changing the original

        """

        msgs = messages if inplace else copy.deepcopy(messages)

        for m in msgs:
            m["content"] = m["content"].format(**values_to_populate)
        
        return msgs


    def __str__(self):
        return self.vendor



