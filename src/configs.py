from datetime import datetime
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class QueryParam:
    mode: Literal["local", "global"] = "global"
    response_type: str = "Multiple Paragraphs"
    level: int = 2
    top_k: int = 20

    # local search with a max content token size of 12000
    # shared across 3 gragh knowledge source
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33
    local_max_token_for_local_context: int = 4800  # 12000 * 0.4
    local_max_token_for_community_report: int = 3200  # 12000 * 0.27
    local_community_single_one: bool = False

    # global search
    global_min_community_rating: float = 0
    global_max_consider_community: float = 512
    global_max_token_for_community_report: int = 16385  # max token size of gpt3.5 turbo
    global_special_community_map_llm_kwargs: dict = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )


@dataclass
class ConfigParam:
    cache_dir = f"./graphrag_lite_cache_{datetime.now().strftime('%Y_%m_%d')}"
    default_model_for_tiktoken: str = "gpt-3.5-turbo"
    chunk_token_size: int = 256
    chunk_overlap_token_size: int = 16
    embedding_batch_num: int = 8
    embedding_func_max_async_call: int = 4
    completion_func_max_async_call: int = 8
    summary_max_tokens: int = 512
    model_max_input_token_size: int = 16385  # gpt-3.5-turbo
    model_max_out_token_size: int = 4096  # gpt-3.5-turbo
    max_llm_gleaning: int = 1
    graph_cluster_algorithm: str = "leiden"
    max_graph_cluster_size: int = 10
    graph_cluster_seed: int = 0xDEADBEEF
