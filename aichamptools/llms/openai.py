import copy
import json
import time
import os

from ..core import LLM
from ..utilities import log_message

from openai import OpenAI, AsyncOpenAI, ChatCompletion, APITimeoutError
import tiktoken

from pydub import AudioSegment



class LLMOpenAI(LLM):
    vendor="OpenAI"
    api_hoster="OpenAI"

    models = {
        # from https://openai.com/pricing 2024.06.04
        "gpt-4o": { "pricing": {"input_tokens": 0.005, "output_tokens": 0.015}, "context_window": 128000 },
        "gpt-4o-2024-05-13": { "pricing": {"input_tokens": 0.005, "output_tokens": 0.015}, "context_window": 128000 },
        "gpt-3.5-turbo-0125": { "pricing": {"input_tokens": 0.0005, "output_tokens": 0.0015}, "context_window": 16000 },
        "gpt-3.5-turbo-instruct": { "pricing": {"input_tokens": 0.0015, "output_tokens": 0.002}, "context_window": 4000 },
        "text-embedding-3-small": { "pricing": {"input_tokens": 0.02}, "context_window": 0 },
        "text-embedding-3-large": { "pricing": {"input_tokens": 0.13}, "context_window": 0 },
        "ada-v2": { "pricing": {"input_tokens": 0.10}, "context_window": 0 },
        "gpt-3.5-turbo-fine-tuning": { "pricing": {"training": 0.008, "input_tokens": 0.003, "output_tokens": 0.006}, "context_window": 0 },
        "davinci-002-fine-tuning": { "pricing": {"training": 0.006, "input_tokens": 0.012, "output_tokens": 0.012}, "context_window": 0 },
        "babbage-002-fine-tuning": { "pricing": {"training": 0.0004, "input_tokens": 0.0016, "output_tokens": 0.0016}, "context_window": 0 },
        "DALL-E 3": { "pricing": {"standard_1024x1024": 0.040, "standard_1024x1792": 0.080, "HD_1024x1024": 0.080, "HD_1024x1792": 0.120}, "context_window": 0 },
        "DALL-E 2": { "pricing": {"1024x1024": 0.020, "512x512": 0.018, "256x256": 0.016}, "context_window": 0 },
        "Whisper": { "pricing": {"input_tokens": 0.006}, "context_window": 0 },
        "TTS": { "pricing": {"input_tokens": 0.015}, "context_window": 0 },
        "TTS HD": { "pricing": {"input_tokens": 0.030}, "context_window": 0 },
        "gpt-4-turbo": { "pricing": {"input_tokens": 0.01, "output_tokens": 0.03}, "context_window": 128000 },
        "gpt-4-turbo-2024-04-09": { "pricing": {"input_tokens": 0.01, "output_tokens": 0.03}, "context_window": 128000 },
        "gpt-4": { "pricing": {"input_tokens": 0.03, "output_tokens": 0.06}, "context_window": 8000 },
        "gpt-4-32k": { "pricing": {"input_tokens": 0.06, "output_tokens": 0.12}, "context_window": 32000 },
        "gpt-4-0125-preview": { "pricing": {"input_tokens": 0.01, "output_tokens": 0.03}, "context_window": 128000 },
        "gpt-4-1106-preview": { "pricing": {"input_tokens": 0.01, "output_tokens": 0.03}, "context_window": 128000 },
        "gpt-4-vision-preview": { "pricing": {"input_tokens": 0.01, "output_tokens": 0.03}, "context_window": 0 },

        "no-pricing": { "pricing": {"input_tokens": 0, "output_tokens": 0}}
    }


    def __init__(self, api_key, api_url=None, vendor=None, api_hoster=None, log_on=True):

        super().__init__(log_on=log_on)

        self.log_on = log_on if log_on is not None else self.log_on

        self.api_key = api_key
        self.api_url = api_url
        self.client = OpenAI(api_key=self.api_key)
        self.client_async = AsyncOpenAI(api_key=self.api_key)

    def num_tokens(self, text, encoding_name):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens

    def create_completion(self, llm_params=None, messages=None, model=None, output_v=0.02):
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
        if not llm_params:
            llm_params = {}
        if model:
            llm_params["model"] = model
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

            input_tokens = llm_response_original["usage"]["prompt_tokens"]
            output_tokens = llm_response_original["usage"]["completion_tokens"]
            usage = {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost": self.execution_cost(model=llm_params["model"], llm_usage={"input_tokens": input_tokens, "output_tokens": output_tokens})}

            llm_response["unified"] = {
                "message": llm_response_original["choices"][0]["message"]["content"],
                "role": llm_response_original["choices"][0]["message"]["role"],
                "llm_params": llm_params,
                "usage": { **usage, "cost": self.execution_cost(model=llm_params["model"], llm_usage=usage) },
                # "usage": llm_response_original["usage"]
            }

        log_message(self.logger, "info", self, f"""RETURNING: {json.dumps(llm_response,indent=4)}""")

        return llm_response

    async def create_completion_async(self, llm_params, messages, output_v=0.02):
        log_message(self.logger, "info", self, f"""START ASYNC""")
        log_message(self.logger, "info", self, f"""INPUT: llm_params: {llm_params}""")
        log_message(self.logger, "info", self, f"""INPUT: messages: {json.dumps(messages,indent=4)}""")

        full_content = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            stream = await self.client_async.chat.completions.create(
                stream=True,
                **llm_params,
                messages=messages
            )
            start_time = time.time()
            first_token_time = None
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    if not first_token_time:
                        first_token_time = time.time()
                    content = chunk.choices[0].delta.content
                    full_content += content
                    yield {"type": "content", "data": content}
                if chunk.usage:
                    usage["prompt_tokens"] = chunk.usage.prompt_tokens
                    usage["completion_tokens"] = chunk.usage.completion_tokens
                    usage["total_tokens"] = chunk.usage.total_tokens
            end_time = time.time()

            time_to_1st_token = first_token_time - start_time
            llm_generation_time = end_time - start_time

            llm_response = {
                "id": f"chatcmpl-{time.time()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": llm_params.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": usage
            }

            llm_response["usage"]["time_to_1st_token"] = time_to_1st_token
            llm_response["usage"]["generation_time"] = llm_generation_time

            if output_v == 0.02:
                llm_response_original = copy.deepcopy(llm_response)
                final_response = {}
                final_response["original"] = llm_response_original

                input_tokens = llm_response_original["usage"]["prompt_tokens"]
                output_tokens = llm_response_original["usage"]["completion_tokens"]
                usage = {**llm_response["usage"], "input_tokens": input_tokens, "output_tokens": output_tokens, "cost": self.execution_cost(model=llm_params["model"], llm_usage={"input_tokens": input_tokens, "output_tokens": output_tokens})}

                final_response["unified"] = {
                    "message": full_content,
                    "role": "assistant",
                    "llm_params": llm_params,
                    "usage": { **usage, "cost": self.execution_cost(model=llm_params["model"], llm_usage=usage) },
                }
            else:
                final_response = llm_response

            yield {"type": "final", "data": final_response}

        except Exception as e:
                log_message(self.logger, "error", self, f"""Error while trying to generate a completion: {e}""")
                yield {"type": "error", "data": str(e)}



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


