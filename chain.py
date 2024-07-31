from langchain.llms.base import LLM
from moa import moa_generate
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import logging

class moaChat(LLM):
    # max_token: int = 512
    history = []
    api_secret: str = ""

    def __init__(self):
        super().__init__()
        print("construct MOA")

    @property
    def _llm_type(self) -> str:
        return "MOA"
    def moa_completion(self, messages):
        return moa_generate(messages)

    # Override _call method to use API for model inference
    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            callbacks: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        try:
            logging.info("Making API call to Together endpoint.")
            messages = prompt
            # logging.info(f"input_prompt{prompt}")
            response = self.moa_completion(messages)
            logging.info(f"moa_completion response: {response}")
        except Exception as e:
            logging.error(f"Error in TogetherLLM _call: {e}", exc_info=True)
            raise

        return response


