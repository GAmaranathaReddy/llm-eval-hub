import requests
from src.utils.sap_utils import get_oauth_token

class GoogleClient:

    def call(self, deployment_id, model_name, messages):
        url = f"https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/{deployment_id}/models/{model_name}-001:generateContent"
        auth_token = get_oauth_token()
        headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json", "AI-Resource-Group": "66edbc6b-cd6a-4bd9-8656-1ac1bdfc6642"}
        payload = {
            "contents": [
                {
                    "role": "USER",
                    "parts": {
                        "text": messages
                    }
                }
            ]
        }
        response = requests.post(url, json=payload, headers=headers, timeout=50)
        res = response.json()
        generated_text = res.get("candidates")[0].get("content").get("parts")[0].get("text")
        return generated_text