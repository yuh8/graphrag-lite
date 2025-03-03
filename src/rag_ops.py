import asyncio
import json
from .configs import QueryParam
from .default_storage_utils import (
    TextChunkSchema,
    CommunitySchema,
    BaseGraphStorage,
    BaseVectorStorage,
    BaseKVStorage,
)
from .prompt import PROMPTS
from .llm_utils import get_completion_response_from_genai_svc
from .graph_ops import (
    get_community_reports_from_similar_entities,
    get_text_chunks_from_similar_entities,
    get_edge_data_from_similar_entities,
)
from .misc_utils import truncate_list_by_token_size, list_of_list_to_csv_format_str
from .log_utils import display_info, display_warning


async def _get_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
):
    """
    Use vector search to find the most related entities to the query
    and then combine the following entity contexts to form the search context for llm
        1. entity context
        2. community context
        3. edge context
        4. text chunk context

    note that excueting await sequentially without gather is the same as synchronous execution
    """
    results = await entities_vdb.query(query, top_k=QueryParam.top_k)
    breakpoint()
    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        display_warning(
            "Some nodes data are missing, potentially damaged node data storage"
        )
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]
    related_community_contexts = await get_community_reports_from_similar_entities(
        node_datas, community_reports
    )
    related_text_chunks = await get_text_chunks_from_similar_entities(
        node_datas, text_chunks_db, knowledge_graph_inst
    )
    related_edge_contexts = await get_edge_data_from_similar_entities(
        node_datas, knowledge_graph_inst
    )
    display_info(
        f"Using {len(node_datas)} entites, {len(related_community_contexts)} communities, {len(related_edge_contexts)} relations, {len(related_text_chunks)} text chunks for RAG"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],  # node degree,
            ]
        )
    entities_context = list_of_list_to_csv_format_str(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "weight", "rank"]
    ]
    for i, e in enumerate(related_edge_contexts):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],  # edge degree = sum of connected nodes' degrees
            ]
        )
    relations_context = list_of_list_to_csv_format_str(relations_section_list)

    communities_section_list = [["id", "content"]]
    for i, c in enumerate(related_community_contexts):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv_format_str(communities_section_list)

    text_chunks_section_list = [["id", "content"]]
    for i, t in enumerate(related_text_chunks):
        text_chunks_section_list.append([i, t["content"]])
    text_chunks_context = list_of_list_to_csv_format_str(text_chunks_section_list)
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_chunks_context}
```
"""


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
) -> str:
    """
    vector search to determine the most similar entities/nodes
    """
    context = await _get_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
    )
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["local_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=QueryParam.response_type
    )
    response = await get_completion_response_from_genai_svc(
        query,
        system_prompt=sys_prompt,
    )
    return response


async def get_summary_points_wrt_query_per_community(
    query: str,
    communities_data: list[CommunitySchema],
):
    community_groups = []
    # truncate all community data to be within token limit
    while len(communities_data):
        this_group = truncate_list_by_token_size(
            communities_data,
            key=lambda x: x["report_string"],
            max_token_size=QueryParam.global_max_token_for_community_report,
        )
        community_groups.append(this_group)
        communities_data = communities_data[len(this_group) :]

    async def _process(community_truncated_datas: list[CommunitySchema]) -> dict:
        communities_section_list = [["id", "content", "rating", "importance"]]
        for i, c in enumerate(community_truncated_datas):
            communities_section_list.append(
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", 0),
                    c["occurrence"],
                ]
            )
        community_context = list_of_list_to_csv_format_str(communities_section_list)
        sys_prompt_temp = PROMPTS["global_map_rag_points"]
        sys_prompt = sys_prompt_temp.format(context_data=community_context)
        response = await get_completion_response_from_genai_svc(
            prompt=query, system_prompt=sys_prompt
        )
        response = json.loads(response)
        return response.get("points", [])

    display_info(f"Grouping to {len(community_groups)} groups for global search")
    responses = await asyncio.gather(*[_process(c) for c in community_groups])
    return responses


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    community_reports: BaseKVStorage[CommunitySchema],
) -> str:
    """
    No vector search, simply provide a globle response to query based on all community data
    """
    community_schema = await knowledge_graph_inst.community_schema()
    community_schema = {
        k: v for k, v in community_schema.items() if v["level"] <= QueryParam.level
    }
    if not len(community_schema):
        return PROMPTS["fail_response"]

    # sorted converts a dictionary into a list of tuples
    # where the first element of the tuple is dictionary key
    # and the second is dictionary value
    sorted_community_schemas = sorted(
        community_schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )
    sorted_community_schemas = sorted_community_schemas[
        : QueryParam.global_max_consider_community
    ]
    community_datas = await community_reports.get_by_ids(
        [k[0] for k in sorted_community_schemas]
    )
    community_datas = [c for c in community_datas if c is not None]
    community_datas = [
        c
        for c in community_datas
        if c["report_json"].get("rating", 0) >= QueryParam.global_min_community_rating
    ]
    community_datas = sorted(
        community_datas,
        key=lambda x: (x["occurrence"], x["report_json"].get("rating", 0)),
        reverse=True,
    )
    display_info(f"Retrieved {len(community_datas)} communities")

    community_summary_points = await get_summary_points_wrt_query_per_community(
        query, community_datas
    )
    final_summary_points = []
    for i, mc in enumerate(community_summary_points):
        for point in mc:
            if "description" not in point:
                continue
            final_summary_points.append(
                {
                    "analyst": i,
                    "answer": point["description"],
                    "score": point.get("score", 1),
                }
            )
    final_summary_points = [p for p in final_summary_points if p["score"] > 0]
    if not len(final_summary_points):
        return PROMPTS["fail_response"]
    final_summary_points = sorted(
        final_summary_points, key=lambda x: x["score"], reverse=True
    )
    final_summary_points = truncate_list_by_token_size(
        final_summary_points,
        key=lambda x: x["answer"],
        max_token_size=QueryParam.global_max_token_for_community_report,
    )
    points_context = []
    for dp in final_summary_points:
        points_context.append(
            f"""----Analyst {dp['analyst']}----
                Importance Score: {dp['score']}
                {dp['answer']}
             """
        )
    points_context = "\n".join(points_context)
    sys_prompt_temp = PROMPTS["global_reduce_rag_response"]
    response = await get_completion_response_from_genai_svc(
        query,
        sys_prompt_temp.format(
            report_data=points_context, response_type=QueryParam.response_type
        ),
    )
    return response
