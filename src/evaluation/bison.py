import requests
from src.utils.sap_utils import get_oauth_token

class BisonClient:

    def call(self, deployment_id, model_name, messages):
        url = f"https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/{deployment_id}/models/{model_name}@002:predict"
        auth_token = get_oauth_token()
        headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json", "AI-Resource-Group": "66edbc6b-cd6a-4bd9-8656-1ac1bdfc6642"}
        payload = {
            "instances": [
                {
                    "messages": [
                        {
                            "author": "user",
                            "content": messages
                        }
                    ]
                }
            ]
        }
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        res = response.json()
        generated_text = res.get("predictions")[0].get("candidates")[0].get("content")
        return generated_text