import boto3
import json
from typing import Dict, Any, List, Optional

# Model inference profile ARNs
CLAUDE_37_ARN = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
DEEPSEEK_R1_ARN = "us.deepseek.r1-v1:0"
LLAMA3_ARN = "us.meta.llama3-3-70b-instruct-v1:0"
MISTRAL_LARGE_ARN = "mistral.mistral-large-2402-v1:0"
MIXTRAL_8X7B_ARN = "mistral.mixtral-8x7b-instruct-v0:1"
HAIKU_35_ARN = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

class BedrockClient:
    """Client for interacting with various Amazon Bedrock text generation models."""

    def __init__(self, region_name: str = "us-east-1", profile_name: Optional[str] = None):
        session = boto3.Session(profile_name=profile_name, region_name=region_name) if profile_name \
                  else boto3.Session(region_name=region_name)
        self.bedrock_runtime = session.client("bedrock-runtime")

    def _invoke_model(self, model_id: str, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal helper to invoke a Bedrock model and return the parsed JSON response.
        """
        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
            raw = response.get("body").read()
            return json.loads(raw.decode('utf-8'))
        except Exception as e:
            return {"error": str(e)}

    def generate_claude(self,
                        prompt: str,
                        max_tokens: int = 1024,
                        temperature: float = 1,
                        top_p: Optional[float] = None,
                        top_k: Optional[int] = None,
                        stop_sequences: Optional[List[str]] = None,
                        system_prompt: Optional[str] = None,
                        MODEL_ID = CLAUDE_37_ARN) -> str:
        """
        Invoke Claude 3.7 and return the generated text.
        """
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if stop_sequences:
            body["stop_sequences"] = stop_sequences
        if system_prompt:
            body["system"] = system_prompt

        resp = self._invoke_model(MODEL_ID, body)
        if "error" in resp:
            return f"Error: {resp['error']}"

        parts = [item.get("text", "") for item in resp.get("content", []) if item.get("type") == "text"]
        return "".join(parts)

    def generate_deepseek(self,
                           prompt: str,
                           max_tokens: int = 1024,
                           temperature: float = 1,
                           top_p: Optional[float] = None,
                           stop_sequences: Optional[List[str]] = None) -> str:
        """
        Invoke DeepSeek R1 and return the generated text.
        """
        formatted = f"<｜begin▁of▁sentence｜><｜User｜>{prompt}<｜Assistant｜>"
        body = {"prompt": formatted, "max_tokens": max_tokens, "temperature": temperature}
        if top_p is not None:
            body["top_p"] = top_p
        if stop_sequences:
            body["stop"] = stop_sequences

        resp = self._invoke_model(DEEPSEEK_R1_ARN, body)
        if "error" in resp:
            return f"Error: {resp['error']}"

        choices = resp.get("choices", [])
        return choices[0].get("text", "") if choices else ""

    def generate_llama(self,
                       prompt: str,
                       max_tokens: int = 1024,
                       temperature: float = 1,
                       top_p: Optional[float] = None,
                       top_k: Optional[int] = None,
                       stop_sequences: Optional[List[str]] = None) -> str:
        """
        Invoke Llama 3 and return the generated text.
        """
        body = {"prompt": prompt + " <|end|>", "max_gen_len": max_tokens, "temperature": temperature}
        if top_p is not None:
            body["top_p"] = top_p
        if top_k is not None:
            body["top_k"] = top_k
        if stop_sequences:
            body["stop"] = stop_sequences

        resp = self._invoke_model(LLAMA3_ARN, body)
        if "error" in resp:
            return f"Error: {resp['error']}"

        return resp.get("generation", "")

    def generate_mistral_large(self,
                               prompt: str,
                               max_tokens: int = 1024,
                               temperature: float = 1,
                               top_p: Optional[float] = 0.9,
                               top_k: Optional[int] = 50) -> str:
        """
        Invoke Mistral Large and return the generated text.
        """
        formatted = f"<s>[INST] {prompt} [/INST]"
        body = {"prompt": formatted, "max_tokens": max_tokens, "temperature": temperature}

        resp = self._invoke_model(MISTRAL_LARGE_ARN, body)
        if "error" in resp:
            return f"Error: {resp['error']}"

        outputs = resp.get("outputs", [])
        return outputs[0].get("text", "") if outputs else ""

    def generate_mixtral(self,
                         prompt: str,
                         max_tokens: int = 1024,
                         temperature: float = 1,
                         top_p: Optional[float] = 0.9,
                         top_k: Optional[int] = 50) -> str:
        """
        Invoke Mixtral 8x7B and return the generated text.
        """
        formatted = f"<s>[INST] {prompt} [/INST]"
        body = {"prompt": formatted, "max_tokens": max_tokens, "temperature": temperature}

        resp = self._invoke_model(MIXTRAL_8X7B_ARN, body)
        if "error" in resp:
            return f"Error: {resp['error']}"

        outputs = resp.get("outputs", [])
        return outputs[0].get("text", "") if outputs else ""



if __name__ == "__main__":
    client = BedrockClient()
    prompt = "What is the capital of France?"
    print("\n=== Single-model generation tests ===")
    print("[HAIKU 3.5]:\n    ", client.generate_claude(prompt, MODEL_ID=HAIKU_35_ARN))
    print("[Claude 3.7]:\n    ", client.generate_claude(prompt))
    print("[DeepSeek R1]:\n    ", client.generate_deepseek(prompt))
    print("[Llama 3]:\n    ", client.generate_llama(prompt))
    print("[Mistral Large]:\n    ", client.generate_mistral_large(prompt))
    print("[Mixtral 8x7B]:\n    ", client.generate_mixtral(prompt))