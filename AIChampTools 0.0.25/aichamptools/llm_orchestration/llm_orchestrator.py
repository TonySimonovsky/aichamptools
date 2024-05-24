# llm_orchestrator.py

from ..llms import *
from ..utilities import log_message, setup_logging
from datetime import datetime

class LLMOrchestrator:

    logger = setup_logging()

    def execute_generation(self, messages_template, datapoints, llm, llm_params, number_of_completions):
        """
        Generates completions for given messages templates and datapoints using the specified LLM.

        :param messages_template: Either a list of dicts, each containing template information with placeholders for data,
                                  or a dict with key "messages_template" which contains the messages template themselves.
        :param datapoints: list of dicts, each containing data to be used in the template.
        :param llm: an instance of an LLM client.
        :param llm_params: parameters specific to the LLM being used.
        :param number_of_completions: number of completions to generate per llm-datapoint.
        :return: list of dicts containing the completions.
        """
        completions = []


        # Normalize messages_template format
        if isinstance(messages_template, list):
            messages_template = {'messages_template': messages_template}

        for datapoint in datapoints:
            # Normalize datapoint format
            if 'data' not in datapoint:
                datapoint = {'data': datapoint}
            
            # Construct the full message by replacing placeholders in all parts of the template
            messages_copy = []
            for template in messages_template['messages_template']:
                # Create a copy of the template and substitute values
                template_copy = template.copy()
                template_copy['content'] = template['content'].format_map(datapoint['data'])
                messages_copy.append(template_copy)
            
            messages_template_copy = messages_template.copy()
            messages_template_copy['messages_template'] = messages_copy

            for _ in range(number_of_completions):
                try:
                    completion = llm.create_completion(llm_params, messages_copy)
                    completion["created_at"] = datetime.now().isoformat()
                    completion["messages"] = messages_copy
                    completions.append({
                        'execution_params': { 'vendor': llm.vendor, 'llm_params': llm_params },
                        'messages_template': messages_template_copy,
                        'datapoint': datapoint,
                        'completion': completion
                    })
                except Exception as e:
                    log_message(self.logger, "error", self, f"""couldn't generate completion for datapoint {datapoint['data']}: {e}""")

        log_message(self.logger, "info", self, f"""Returning {len(completions)} completions""")

        return completions
