import requests
from src.utils.sap_utils import get_oauth_token

class BedrockClient:

    def call(self, deployment_id, model_name, messages):
        url = f"https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/{deployment_id}/invoke"
        auth_token = get_oauth_token()
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
            "AI-Resource-Group": "66edbc6b-cd6a-4bd9-8656-1ac1bdfc6642"
        }

        payload = self._create_payload(model_name, messages)
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        res = response.json()
        generated_text = self._extract_generated_text(res, model_name)
        return generated_text

    def _create_payload(self, model_name, messages):
        if "amazon" in model_name:
            return {
                "inputText": messages,
                "textGenerationConfig": {
                    "maxTokenCount": 4096,
                    "temperature": 0
                }
            }
        elif "anthropic" in model_name:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": messages
                    }
                ]
            }
        else:
            raise ValueError("Unsupported model name")

    def _extract_generated_text(self, response, model_name):
        if "amazon" in model_name:
            return response.get('results')[0].get('outputText')
        elif "anthropic" in model_name:
            return response.get("content")[0].get("text")
        else:
            raise ValueError("Unsupported model name")