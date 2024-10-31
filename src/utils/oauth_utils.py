import requests
import cachetools.func



@cachetools.func.ttl_cache(maxsize=32, ttl=(6 * 60 * 60))
def get_oauth_token() -> str:
    """Generate an OAuth token for vLLM Custom Provider.

    Returns:
        str: OAuth token
    """
    print("Getting OAuth token.")
    client_id = "Your Client ID"
    client_secret = "Your Client Secret"
    auth_endpoint = "Your Auth Endpoint"
    resp = requests.post(
        auth_endpoint,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=3600,
    )
    return resp.json().get("access_token")
