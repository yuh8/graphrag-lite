import asyncio
import numpy as np
from functools import wraps
from aiohttp import ClientResponse, ClientSession
from aiohttp_retry import RetryClient, ExponentialRetry
from .sand_utlis import get_sandtoken
from .configs import ConfigParam
from .log_utils import display_info
from .misc_utils import try_get_config_from_env

BASE_URL = try_get_config_from_env("GENAI_URL")


def limit_async_func_call(max_num_coroutines: int) -> callable:
    """Concurrency control for embedding function calling"""

    display_info(f"Concurrent asynchronous calls limited to {max_num_coroutines}")
    semaphore = asyncio.Semaphore(max_num_coroutines)

    def final_decro(func):
        @wraps(func)
        async def wait_func(*args, **kwargs):
            async with semaphore:
                result = await func(*args, **kwargs)
            return result

        return wait_func

    return final_decro


async def _fetch_with_retry(url: str, headers: dict, data: dict) -> ClientResponse:
    retry_options = ExponentialRetry(
        attempts=5, start_timeout=1, factor=2, statuses={429}
    )
    async with ClientSession() as client:
        retry_client = RetryClient(client)
        resp = await retry_client.post(
            url,
            json=data,
            headers=headers,
            raise_for_status=False,
            retry_options=retry_options,
        )
        resp_content = await resp.json()
        resp_status = resp.status
        await retry_client.close()
    if resp_status == 401:
        return resp_content, resp_status
    if "embeddings" in url:
        resp_content = [np.array(v["embedding"]) for v in resp_content["data"]]
    return resp_content, resp_status


@limit_async_func_call(max_num_coroutines=ConfigParam.completion_func_max_async_call)
async def get_completion_response_from_genai_svc(
    prompt: str, system_prompt: str = None, history_messages: list[dict] = []
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    sand_token = get_sandtoken()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {sand_token}",
        # "X-Coupa-Tenant": "graphrag-lite",
        # "X-Coupa-Application": try_get_config_from_env("GENAI_APPLICATION_ID"),
    }

    url = BASE_URL + "/v1/openai/chat/completions"
    data = {"messages": messages, "temperature": 1e-10, "model": "gpt-4o"}
    resp_txt, status_code = await _fetch_with_retry(url, headers, data)
    if status_code == 401:
        sand_token = get_sandtoken(fetch_new=True)
        headers["Authorization"] = f"Bearer {sand_token}"
        resp_txt, _ = await _fetch_with_retry(url, headers, data)

    return resp_txt["choices"][0]["message"]["content"]


@limit_async_func_call(max_num_coroutines=ConfigParam.embedding_func_max_async_call)
async def get_embedding_response_from_genai_svc(data: list[str]) -> ClientResponse:
    sand_token = get_sandtoken()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {sand_token}",
        # "X-Coupa-Tenant": "graphrag-lite",
        # "X-Coupa-Application": try_get_config_from_env("GENAI_APPLICATION_ID"),
    }

    url = BASE_URL + "/v1/openai/embeddings"
    data = {
        "temperature": 1e-10,
        "deployment_id": "embedding-dev",
        "model": "text-embedding-ada-002",
        "input": data,
    }
    batch_embedding, status_code = await _fetch_with_retry(url, headers, data)
    if status_code == 401:
        sand_token = get_sandtoken(fetch_new=True)
        headers["Authorization"] = f"Bearer {sand_token}"
        batch_embedding, _ = await _fetch_with_retry(url, headers, data)

    return batch_embedding
