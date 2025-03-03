import os
import json
import base64
import requests
from .log_utils import display_error
from .misc_utils import try_get_config_from_env, create_folder


def save_access_string(access_string: str, filename: str = "access_cache.json") -> None:
    create_folder("cache")
    cache = {}
    cache["access_string"] = access_string

    with open("cache/" + filename, "w") as f:
        json.dump(cache, f)


def get_cached_access_string(filename: str = "access_cache.json"):
    if os.path.exists("cache/" + filename):
        with open("cache/" + filename, "r") as f:
            cache = json.load(f)
        return cache.get("access_string")
    return None


class SAND:
    def __init__(self, id: str, secret: str, scope: str, env: str = "dev") -> None:
        self.id = id
        self.secret = secret
        self.scope = scope
        self.env = env
        self._token = None
        self.sand_url = f"https://sand-{env}.io.coupadev.com"

    def encode_client_credentials(self, client_id: str, client_secret: str) -> str:
        credentials_str = f"{client_id}:{client_secret}"
        encoded_credentials = base64.b64encode(credentials_str.encode()).decode()
        return encoded_credentials

    def _get_token(self) -> str:
        url = f"{self.sand_url}/oauth2/token"
        data = {"grant_type": "client_credentials", "scope": self.scope}
        credential = self.encode_client_credentials(self.id, self.secret)
        headers = {"Authorization": f"Basic {credential}"}
        response = requests.post(url, data=data, headers=headers)
        try:
            return response.json()["access_token"]
        except Exception as e:
            print(f"Unable to get access token: {e}")
            raise e

    @property
    def token(self) -> str:
        if not self._token:
            self._token = self._get_token()
        return self._token


def get_sandtoken(fetch_new: bool = False) -> str:
    try:
        SAND_CLIENT_ID = try_get_config_from_env("SAND_CLIENT_ID")
        SAND_CLIENT_SECRET = try_get_config_from_env("SAND_CLIENT_SECRET")
        SAND_SCOPES = try_get_config_from_env("SAND_SCOPES")
        token = get_cached_access_string()
        if (not token) or fetch_new:
            sand = SAND(SAND_CLIENT_ID, SAND_CLIENT_SECRET, SAND_SCOPES)
            save_access_string(sand.token)
        else:
            return token
    except Exception as e:
        display_error(f"Failed to get sand token due to: {e}")
        return "No token"
    return sand.token
