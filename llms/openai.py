import copy
import json
import time
import os

from ..core import LLM
from ..utilities import log_message

from openai import OpenAI, ChatCompletion, APITimeoutError
import tiktoken

from pydub import AudioSegment



class LLMOpenAI(LLM):
    vendor="OpenAI"

    models = {
        # from https://openai.com/pricing on 2023.11.16
        "gpt-4-1106-preview": { "pricing": {"prompt_tokens": 0.01, "completion_tokens": 0.03}, "context_window": 128000 },
        "gpt-4-1106-vision-preview": { "pricing": {"prompt_tokens": 0.01, "completion_tokens": 0.03}},
        "gpt-4": { "pricing": {"prompt_tokens": 0.03, "completion_tokens": 0.06}, "context_window": 16000 },
        "gpt-4-32k": { "pricing": {"prompt_tokens": 0.06, "completion_tokens": 0.12}, "context_window": 32000},
        "gpt-3.5-turbo-1106": { "pricing": {"prompt_tokens": 0.001, "completion_tokens": 0.0020}, "context_window": 16000 },

        # guesses on 2023.11.16
        "gpt-3.5-turbo": { "pricing": {"prompt_tokens": 0.001, "completion_tokens": 0.0020}, "context_window": 4096 },
        "gpt-3.5-turbo-0613": { "pricing": {"prompt_tokens": 0.001, "completion_tokens": 0.0020}},

        "no-pricing": { "pricing": {"prompt_tokens": 0, "completion_tokens": 0}}
    }


    def __init__(self, api_key, log_on=True):

        # print(f"({self}) 0 TMP log_on: {log_on}")

        super().__init__(log_on=log_on)

        # print(f"({self}) 1 TMP log_on: {log_on}, self.log_on: {self.log_on}")

        self.log_on = log_on if log_on is not None else self.log_on

        # print(f"({self}) 2 TMP log_on: {log_on}, self.log_on: {self.log_on}")

        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)


    def num_tokens(self, text, encoding_name):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens


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


    def transcribe(self, file_path, model="whisper-1"):

        log_message(self.logger, "info", self, f"""START""")
        log_message(self.logger, "info", self, f"""INPUT: file_path: {file_path}""")

        directory_path = os.path.dirname(file_path)

        log_message(self.logger, "info", self, f"""INPUT: directory_path: {directory_path}""")


        try:
            song = AudioSegment.from_file(file_path,"mp3")
        except Exception as e:
            log_message(self.logger, "error", self, f"""Couldn't read the file {file_path}: {e}""")
            return None

        # PyDub handles time in milliseconds
        chunk_length = 10 * 60 * 1000  # 10 minutes
        chunks = [song[i:i + chunk_length] for i in range(0, len(song), chunk_length)]

        full_transcript = ""
        for i, chunk in enumerate(chunks):
            chunk_filepath = f"{directory_path}/chunk_{i}.mp3"
            chunk.export(chunk_filepath, format="mp3")

            # Check if the chunk file size is less than 25 MB
            if os.path.getsize(chunk_filepath) < 25 * 1024 * 1024:
                with open(chunk_filepath, "rb") as file:
                    log_message(self.logger, "info", self, f"""Starting chunk {i} ('{chunk_filepath}') transcription""")

                    try:
                        transcript = self.client.audio.transcriptions.create(
                            model=model,
                            file=file
                        )
                        n = 200

                        # print(f"TMP ", transcript)

                        log_message(self.logger, "info", self, f"""Chunk {i} of '{file_path}' transcribed successfully. First {n} chars: {transcript.text[0:n]}... ({len(transcript.text)-n} more)""")
                    except Exception as e:
                        log_message(self.logger, "error", self, f""""Couldn't transcribe the chunk {i} of '{file_path}': {e}""")
                    
                    full_transcript += transcript.text

            # Remove the chunk file after transcribing
            os.remove(chunk_filepath)


        log_message(self.logger, "info", self, f"""RETURN""")

        return full_transcript


