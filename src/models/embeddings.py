from utils.oauth_utils import get_oauth_token
import requests
import numpy as np

def generate_embeddings(text_list, deployment_id):
    endpoint = f"https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/{deployment_id}/embeddings?api-version=2023-05-15"
    auth_token = get_oauth_token()
    headers = {"Authorization": f"Bearer {auth_token}", "Content-Type": "application/json", "AI-Resource-Group": "66edbc6b-cd6a-4bd9-8656-1ac1bdfc6642"}
    test_input = { "model" : "text-embedding-ada-002", "input" : text_list }
    response = requests.post(endpoint, headers=headers, json=test_input)
    res = response.json().get("data")[0].get("embedding")
    return np.array(res)
