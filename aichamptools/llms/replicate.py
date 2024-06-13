import copy
import json
import time
import os
import replicate
import re

from ..core import LLM
from ..utilities import log_message


class OnReplicate(LLM):
    vendor=""
    api_hoster="Replicate"

    models = {
    }


    def __init__(self, api_key, api_url=None, vendor=None, api_hoster=None, log_on=True):

        super().__init__(log_on=log_on)

        self.log_on = log_on if log_on is not None else self.log_on

        self.api_key = api_key
        self.api_url = api_url
        self.client = replicate.Client(api_token=self.api_key)



    def transcribe(self, file_path=None, link=None, model=None):
        log_message(self.logger, "info", self, f"""START""")
        log_message(self.logger, "info", self, f"""INPUT: file_path: {file_path}""")

        input = {
            "file": link,
            "prompt": "",
            "language": "en",
            "group_segments": True,
            "offset_seconds": 0,
            "transcript_output_format": "both"
        }
        output = self.client.run(
            "thomasmol/whisper-diarization:b9fd8313c0d492bf1ce501b3d188f945389327730773ec1deb6ef233df6ea119",
            input=input
        )

        transcript_text = ' '.join(segment['text'] for segment in output["segments"])

        log_message(self.logger, "info", self, f"""RETURN""")

        return { "text": transcript_text, "diarised": output }


