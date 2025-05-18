import time
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import uuid
import requests
import json
from pydantic import BaseModel
import os
#from tool import other_tools

class CustomChatGPT(LLM):

    base_url: str
    session: Optional[str]
    parent_message_id: Optional[str]

    @property
    def _llm_type(self) -> str:
        return "custom_chatgpt"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        conversation_id: Optional[str] = None,
        parent_message_id: Optional[str] = None,
    ) -> str:
        # if stop is not None:
        #     raise ValueError("stop kwargs are not permitted.")
        model = "text-davinci-002-render-sha"#gpt-4/text-davinci-002-render-sha
        message_id = str(uuid.uuid4())
        if parent_message_id is None:
            if self.parent_message_id is not None:
                parent_message_id = self.parent_message_id
            else:
                parent_message_id = str(uuid.uuid4())
            # parent_message_id = self.parent_message_id
        if conversation_id is None and self.session is not None:
            conversation_id = self.session
        data = {
            "prompt": prompt,
            "model": model,
            "message_id": message_id,
            "parent_message_id": parent_message_id,
            "stream": False,
            "conversation_id": conversation_id,
        }
        response = requests.post(f"{self.base_url}/api/conversation/talk", json=data)
        response_data = response.text
        response_data = json.loads(response_data)
        parts = response_data['message']['content']['parts']
        response_message = ''.join(parts)
        if stop:
            for stop_value in stop:
                stop_index = response_message.find(stop_value)
                if stop_index != -1:
                    response_message = response_message[:stop_index]
                    break
        if self.session is None:
            self.session = response_data["conversation_id"]
        self.parent_message_id = response_data["message"]['id']
        return response_message

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"base_url": self.base_url}



class FakeLLM(LLM):

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
            self,
            prompt: str = None,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        prompt = '''Action: avg_price
Action Input: SOLUSDT

Action: avg_price
Action Input: BTCUSDT

Action: avg_price
Action Input: ETHUSDT

Action: avg_price
Action Input: BNBUSDT

Action: avg_price
Action Input: XRPUSDT'''
        return prompt

def main():
    llm = CustomChatGPT(base_url="http://127.0.0.1:1079")
    response = llm("Can you introduce yourself?")
    print(response)

if __name__ == "__main__":
    main()