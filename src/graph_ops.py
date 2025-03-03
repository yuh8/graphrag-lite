import re
import json
import asyncio
from collections import Counter, defaultdict
from typing import Union
from .llm_utils import get_completion_response_from_genai_svc
from .configs import ConfigParam, QueryParam
from .misc_utils import (
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    clean_str,
    is_float_regex,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    list_of_list_to_csv_format_str,
)
from .prompt import PROMPTS, GRAPH_FIELD_SEP
from .default_storage_utils import (
    TextChunkSchema,
    SingleEntitySchema,
    SingleEdgeSchema,
    SingleCommunitySchema,
    CommunitySchema,
    BaseGraphStorage,
    BaseVectorStorage,
    BaseKVStorage,
)
from .log_utils import display_warning, display_info
from .misc_utils import compute_mdhash_id


def chunking_by_token_size(
    content: str,
    overlap_token_size=128,
    max_token_size=1024,
) -> list[dict]:
    """
    tokenize all texts and chunkify into chunks of max_token_size
    finally decode all tokens in each chunk back to text
    """
    tiktoken_model = ConfigParam.default_model_for_tiktoken
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def get_entity_or_relation_summary(
    entity_or_relation_name: str, description: str
) -> str:
    """
    for each entity or pair of connected entities, provide a summary for it based on its
    list (or joined list) of decriptions
    """
    llm_max_tokens = ConfigParam.model_max_input_token_size
    tiktoken_model_name = ConfigParam.default_model_for_tiktoken
    summary_max_tokens = ConfigParam.summary_max_tokens

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,  # node or edge pair
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    user_prompt = prompt_template.format(**context_base)
    summary = await get_completion_response_from_genai_svc(prompt=user_prompt)
    return summary


async def format_single_entity_extraction(
    record_attributes: list[str],
    source_text_chunk_key: str,
) -> SingleEntitySchema:
    """
    store properties of a single extracted entity in a dictionary.
    This entity info will be added as a node in a knowledge graph
    """
    if record_attributes[0] != '"entity"' or len(record_attributes) < 4:
        return None
    # will add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = source_text_chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def format_single_edge_extraction(
    record_attributes: list[str],
    chunk_key: str,
) -> SingleEdgeSchema:
    """
    store properties of a single extracted edge between two entities in a dictionary
    The edge info will be added as a edge in the knowledge graph
    """
    if record_attributes[0] != '"relationship"' or len(record_attributes) < 5:
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    # here src_entity and tgt_entity are just the original entity names in str
    # source_id is the original text chunk from which relationshp is derived
    return dict(
        src_entity=source,
        tgt_entity=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def merge_nodes_info_then_upsert(
    entity_name: str, nodes_data: list[dict], knwoledge_graph_inst: BaseGraphStorage
) -> SingleEntitySchema:
    """
    For a node/entity, it could appear in multiple text chunks, so we need to provide
    a comprehensive and a single merged summary of the node entity using LLM.
        1. We collect all containing text chunks and merge them as description
        2. We collect all containing text chunks id and join them with GRAPH_FIELD_SEP
        3. We use LLM to summarize the merged description
        4. Update the node description with the summarized description

    nodes_data is a list of dictionaries with keys: [entity_name, entity_type, entity_desc, chunk_key]
    belonging to the same entity
    """
    existed_entitiy_types = []
    existed_source_ids = []
    existed_description = []

    # check if node already exists
    existed_node = await knwoledge_graph_inst.get_node(entity_name)
    if existed_node is not None:
        existed_entitiy_types.append(existed_node["entity_type"])
        existed_source_ids.extend(
            split_string_by_multi_markers(existed_node["source_id"], [GRAPH_FIELD_SEP])
        )
        existed_description.append(existed_node["description"])

    # sort entity types by their occurrences including the queried node
    # and select the most frequently occuring entity type
    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + existed_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    # remove duplicated description, sort alphabetically and join in a single description string
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + existed_description))
    )
    # join all unique source chunks within which entity is contained
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + existed_source_ids)
    )
    # based on all related text chunks, provide a concise summary for that entity and
    # replace the joined description with LLM summarized description
    description = await get_entity_or_relation_summary(entity_name, description)

    # node data contain the most frequenty entity type for a particular entity
    # also contains summarized description from LLM
    # and contains all source chunk ids
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )

    # add node data to a nx.Graph()
    await knwoledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def merge_edges_info_then_upsert(
    src_entity: str,
    tgt_entity: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
) -> None:
    """
    Simlarily, same edge/connection between entities could appear in multiple text chunks.
    We provide a comprehensive single summary of the edge info using LLM.
    For an edge with source_entity_name and target_entity_name:
        1. we collect all edges containing these two ids
        2. we sum all edge weights as the new weight
        3. we collect all unique source_chunk_ids for this edge as the new source_id str
        4. We collect all descriptions of this edge and summarize these descriptions using llm
        5. we update the edge with new weight, source id and summarized description
    """
    existed_weights = []
    existed_source_ids = []
    existed_description = []
    # if graph has an edge between two entities
    if await knwoledge_graph_inst.has_edge(src_entity, tgt_entity):
        existed_edge = await knwoledge_graph_inst.get_edge(src_entity, tgt_entity)
        existed_weights.append(existed_edge["weight"])
        existed_source_ids.extend(
            split_string_by_multi_markers(existed_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        existed_description.append(existed_edge["description"])

    # sum all edge weigts as the new weight
    weight = sum([dp["weight"] for dp in edges_data] + existed_weights)
    # concat all edge descriptions
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + existed_description))
    )
    # concat all unique source_chunk_ids
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + existed_source_ids)
    )
    for need_insert_id in [src_entity, tgt_entity]:
        # if either source entity or target entity does not exist,
        # we add that entity as a new node and use the same source_chunk_id
        if not (await knwoledge_graph_inst.has_node(need_insert_id)):
            await knwoledge_graph_inst.upsert_node(
                need_insert_id,  # entity_name_value
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    # summarize all descriptions for this edge
    description = await get_entity_or_relation_summary(
        (src_entity, tgt_entity), description
    )

    # add edge data to a nx.Graph()
    await knwoledge_graph_inst.upsert_edge(
        src_entity,
        tgt_entity,
        edge_data=dict(
            weight=weight,
            description=description,
            source_id=source_id,
        ),
    )


def concat_user_assit_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args)
    ]


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    entity_vdb: BaseVectorStorage,
    knwoledge_graph_inst: BaseGraphStorage,
) -> Union[BaseGraphStorage, None]:
    """
    extract asynchronously all entities and their relationships from all doc
    chunks
    """
    # max number of LLM solicitation
    entity_extract_max_gleaning = ConfigParam.max_llm_gleaning

    # list of tuple(chunk_id, TextChunkSchema)
    ordered_chunks = list(chunks.items())

    # prompt to extract all entities and all of their relationships
    # entity tuple starts with "entity" keyword
    # relationship tuple start with "relationship" keyword
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0

    breakpoint()

    # extract entity and their relationships per chunk
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        # refer to already_processed in the outside extract_entities function
        nonlocal already_processed
        chunk_key = chunk_key_dp[0]  # md5 hash id for the chunk content
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await get_completion_response_from_genai_svc(prompt=hint_prompt)

        # history = [{"role": "user", "content": " hint_prompt"},
        #            {"role": "assistant", "content"" final_result"}
        history = concat_user_assit_to_openai_messages(hint_prompt, final_result)
        # the number of times to prompt llm to recover all missing entities
        # this is called gleaning
        for now_glean_index in range(entity_extract_max_gleaning):
            # ask LLM to extact any missing entity or relationship by passing historical
            # conversation to it
            glean_result = await get_completion_response_from_genai_svc(
                prompt=continue_prompt, history_messages=history
            )

            history += concat_user_assit_to_openai_messages(
                continue_prompt, glean_result
            )
            # we append the newly discovered entities and relationships to the final result
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            # prompt llm to return if it thinks THERE ARE STILL MISSING entities to be extracted
            # if NO, terminate and proceed to next phase
            if_loop_result: str = await get_completion_response_from_genai_svc(
                if_loop_prompt, history_messages=history
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        # split the final results in records
        # each record is a tuple starting with either "entity" or "relationship" keyword
        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            # parse out a single tuple record from LLM extraction results
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            # [entity_name, entity_type, entity_desc, chunk_key]
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            # make record_attributes a dict
            if_entities = await format_single_entity_extraction(
                record_attributes, chunk_key
            )

            if if_entities is not None:
                _entity_name = re.sub(
                    r"[^a-zA-Z0-9 ]", "", if_entities["entity_name"].lower()
                )
                _cleaned_content = re.sub(r"[^a-zA-Z0-9 ]", "", content.lower())
                if _entity_name not in _cleaned_content:
                    continue
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            # if the first element of record_attributes is not "entity" but "relationship"
            if_relation = await format_single_edge_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[
                    (if_relation["src_entity"], if_relation["tgt_entity"])
                ].append(if_relation)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(f"{now_ticks} Processed {already_processed} chunks\r", end="", flush=True)
        # return nodes and edges info per text chunk
        return dict(maybe_nodes), dict(maybe_edges)

    # async gather returns a list of tuples
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )

    breakpoint()

    # there could be multiple appearances of the same entity in different
    # text chunks, therefore we initialize node as a dict[str, list] with the
    # default list containing all instances of an entity
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's a undirected graph, sort entity name alphabetically
            # undirected, the order of src and tgr does not matter
            maybe_edges[tuple(sorted(k))].extend(v)

    # same as asyncio.gather(func1, func2, ...)
    # update and insert with LLM summmarized entity description
    all_entities_data = await asyncio.gather(
        *[
            merge_nodes_info_then_upsert(k, v, knwoledge_graph_inst)
            for k, v in maybe_nodes.items()
        ]
    )

    # update and insert with LLM summmarized entity relationship description
    await asyncio.gather(
        *[
            merge_edges_info_then_upsert(k[0], k[1], v, knwoledge_graph_inst)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        display_warning("Failed to extract any entities")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)

    # return a knowledge graph that contains summarized description per node and per edge
    breakpoint()
    return knwoledge_graph_inst


def get_single_community_context_by_using_sub_community_contexts(
    community: SingleCommunitySchema,
    max_token_size: int,
    existed_community_reports: dict[str, CommunitySchema],
) -> tuple[str, int, set, set]:
    """
    Generate a csv context table for a single community report generation
    by using the generated reports from its subcommunities
    """
    # subcommonuties must have report str generated
    all_sub_community_reports = [
        existed_community_reports[k]
        for k in community["sub_communities"]
        if k in existed_community_reports
    ]
    all_sub_community_reports = sorted(
        all_sub_community_reports, key=lambda x: x["occurrence"], reverse=True
    )
    truncated_sub_community_reports = truncate_list_by_token_size(
        all_sub_community_reports,
        key=lambda x: x["report_string"],
        max_token_size=max_token_size,
    )
    sub_fields = ["id", "report", "rating", "importance"]
    sub_community_contexts = list_of_list_to_csv_format_str(
        [sub_fields]
        + [
            [
                i,
                c["report_string"],
                c["report_json"].get("rating", -1),
                c["occurrence"],
            ]
            for i, c in enumerate(truncated_sub_community_reports)
        ]
    )
    existed_nodes = []
    existed_edges = []
    # we also output the nodes and edges of the subcommunities which have community report
    # generated
    for c in truncated_sub_community_reports:
        existed_nodes.extend(c["nodes"])
        existed_edges.extend([tuple(e) for e in c["edges"]])
    return (
        sub_community_contexts,
        len(encode_string_by_tiktoken(sub_community_contexts)),
        set(existed_nodes),
        set(existed_edges),
    )


async def get_single_community_context(
    knwoledge_graph_inst: BaseGraphStorage,
    community: SingleCommunitySchema,
    max_token_size: int = 12000,
    existed_reports: dict[str, CommunitySchema] = {},
) -> str:
    """
    prepare csv context table for generating a single community report

    Inputs:
    community = TypedDict(
        "SingleCommunitySchema",
        {
            "level": int,
            "title": str,
            "edges": list[list[str, str]],
            "nodes": list[str],
            "chunk_ids": list[str],
            "occurrence": float,
            "sub_communities": list[str],
        },
    ). This is a single json containing all nodes and edges names in a community.

    existed_reports is a dictionary containing communities with reports already generated.

    A community json does not contain detailed node and edge info but only their keys/names.

    We can retrieve detailed info from knwoledge_graph_inst.
    """
    # sorted nodes by entity name
    nodes_in_order = sorted(community["nodes"])
    # sorted nodes by src_id + tgt_id names str
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    # get a list of node attributes, each element in the list contains
    # a dictionary of LLM summarized node info
    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )

    # get a list of edge attributes, each element in list contains
    # a dictionary of LLM summarized edge info between src and tgt entities
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )

    # the output csv-like context headers for generating a community report
    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]

    # output csv lines
    # Entities:
    # ```csv
    # id,entity,type,description
    # 5,VERDANT OASIS PLAZA,geo,Verdant Oasis Plaza is the location of the Unity March
    # 6,HARMONY ASSEMBLY,organization,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza
    # ```
    nodes_list_data = [
        [
            i,
            node_name,
            node_data.get(
                "entity_type", "UNKNOWN"
            ),  # dict get method with a default value
            node_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.node_degree(node_name),
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]

    # sorted the output list by node degrees
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    # limit the size of a community nodes by truncating community node descriptions until max token size is reached
    truncated_nodes_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    # similar to above
    # Relationships:
    # ```csv
    # id,source,target,description
    # 37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
    # 38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
    # 39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
    edges_list_data = [
        [
            i,
            edge_name[0],
            edge_name[1],
            edge_data.get("description", "UNKNOWN"),
            await knwoledge_graph_inst.edge_degree(*edge_name),
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]

    # sorted the output list by edge degrees
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    # limit the size of a community edges by truncating edge descriptions until max token size //2 is reached
    # why divided by 2? cause both node and edges data share the same context token limit.
    truncated_egdes_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(truncated_nodes_list_data) or len(
        edges_list_data
    ) > len(truncated_egdes_list_data)

    # If original node context or edge context tokens exceeds the limit and have sub-communities:
    # We truncate nodes and edge context, leading to some information loss. To compensate,
    # 1. We add a sub_community_report_summary to the community context
    # 2. We prioritize truncated nodes and edge data for the node and edge context
    sub_community_report_summary = ""
    if truncated and len(community["sub_communities"]) and len(existed_reports):
        display_info(
            f"Original community {community['title']} node/edge context exceeded token limit, truncation executed."
        )
        sub_community_report_summary, report_size, contain_nodes, contain_edges = (
            get_single_community_context_by_using_sub_community_contexts(
                community, max_token_size, existed_reports
            )
        )
        # exclude entities not in subcommunity entities
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]
        # If truncated, the generated context already contains sub_community_report_summary, so we need to deduct this
        # token consumed from token limit of the node context and edge context
        # //2 because we need to allocate token size for both nodes and edges context
        truncated_nodes_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        truncated_egdes_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
    nodes_context = list_of_list_to_csv_format_str(
        [node_fields] + truncated_nodes_list_data
    )
    edges_context = list_of_list_to_csv_format_str(
        [edge_fields] + truncated_egdes_list_data
    )
    return f"""-----Reports-----
```csv
{sub_community_report_summary}
```
-----Entities-----
```csv
{nodes_context}
```
-----Relationships-----
```csv
{edges_context}
```"""


def _community_report_json_to_str(parsed_output: dict) -> str:
    """refer official graphrag: index/graph/extractors/community_reports"""
    title = parsed_output.get("title", "Report")
    summary = parsed_output.get("summary", "")
    findings = parsed_output.get("findings", [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )
    return f"# {title}\n\n{summary}\n\n{report_sections}"


async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],
    knwoledge_graph_inst: BaseGraphStorage,
) -> None:
    """
    we have already obtained cluster/community of the nodes, and we need to generate a LLM
    summarized report for each community
    """
    community_report_prompt = PROMPTS["community_report"]

    communities_schema = await knwoledge_graph_inst.community_schema()
    # keys are community cluster ids, values are the community reports following SingleCommunitySchema
    community_keys, community_values = list(communities_schema.keys()), list(
        communities_schema.values()
    )
    already_processed = 0

    async def _generate_single_community_report(
        community: SingleCommunitySchema, existed_reports: dict[str, CommunitySchema]
    ):
        """
        Use the previously generated csv table for entities and relationships within
        a community to generate a report for the community
        existed_reports: communities that already have reports generated
        """
        nonlocal already_processed

        # community context follows
        # Text:
        # ```
        # Entities:
        # ```csv
        # id,entity,type,description
        # 5,VERDANT OASIS PLAZA,geo,Verdant Oasis Plaza is the location of the Unity March
        # 6,HARMONY ASSEMBLY,organization,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza
        # ```
        # Relationships:
        # ```csv
        # id,source,target,description
        # 37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March
        # 38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza
        # 39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza
        # 40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at Verdant Oasis Plaza
        # 41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march
        # 43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March
        # ```
        context = await get_single_community_context(
            knwoledge_graph_inst,
            community,
            max_token_size=ConfigParam.model_max_input_token_size,
            existed_reports=existed_reports,
        )
        # context will be sent to LLM for generating a json community report
        prompt = community_report_prompt.format(input_text=context)
        response = await get_completion_response_from_genai_svc(prompt=prompt)
        data = json.loads(response)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} communities\r",
            end="",
            flush=True,
        )
        return data

    # sort by community levels
    levels = sorted(set([c["level"] for c in community_values]), reverse=True)
    display_info(f"Generating community report by levels: {levels}")
    community_datas = {}
    for level in levels:
        this_level_community_keys, this_level_community_values = zip(
            *[
                (k, v)
                for k, v in zip(community_keys, community_values)
                if v["level"] == level
            ]
        )
        this_level_communities_reports = await asyncio.gather(
            *[
                _generate_single_community_report(c, community_datas)
                for c in this_level_community_values
            ]
        )
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json": r,
                    **v,
                }
                for k, r, v in zip(
                    this_level_community_keys,
                    this_level_communities_reports,
                    this_level_community_values,
                )
            }
        )
    await community_report_kv.upsert(community_datas)
    return community_report_kv


async def get_community_reports_from_similar_entities(
    node_datas: list[dict],
    community_reports: BaseKVStorage[CommunitySchema],
):
    """
    For a given list of similar nodes obtained by similarity search,
    find the most frequently assigned community clusters.
    Within each cluster, order community reports by rating.
    Truncated community reports by maximum token size.
    Return truncated list of related community reports
    """
    related_communities = []
    for node_d in node_datas:
        if "clusters" not in node_d:
            continue
        # node_d["clusters"] contains all levels and community clusters
        # that a node belongs to
        related_communities.extend(json.loads(node_d["clusters"]))
    related_community_clusters_below_query_level = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= QueryParam.level
    ]
    related_community_cluster_counts = dict(
        Counter(related_community_clusters_below_query_level)
    )
    # list of community reports
    _related_community_datas = await asyncio.gather(
        *[
            community_reports.get_by_id(k)
            for k in related_community_cluster_counts.keys()
        ]
    )

    # dictionary of community reports where keys are cluster ids
    related_community_datas = {
        k: v
        for k, v in zip(
            related_community_cluster_counts.keys(), _related_community_datas
        )
        if v is not None
    }

    # sort by communicate cluster counts and then within each cluster sort by report rating
    related_community_keys = sorted(
        related_community_cluster_counts.keys(),
        key=lambda k: (
            related_community_cluster_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    # list of (list of community reports) community clusters sorted by cluster counts
    truncated_related_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=QueryParam.local_max_token_for_community_report,
    )
    if QueryParam.local_community_single_one:
        truncated_related_community_reports = truncated_related_community_reports[:1]
    return truncated_related_community_reports


async def get_text_chunks_from_similar_entities(
    node_datas: list[dict],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    # list of (list of md5 doc chunk ids) for every similar node
    text_chunk_ids = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    # list of (list of edge tuples) for the similar nodes
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )

    # get the first-order neighbours for all similar nodes
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)

    # get all nodes data for all one-hop neighours
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # get a dictionary of one-hop entity_name: entity_source_chunk_ids
    all_one_hop_text_chunks_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_chunks_lookup = {}

    # for each text chunk of the similar nodes
    # if the text chunk also appears in a neighbour node chunks, then increment relationship by 1
    for index, (this_text_chunk_ids, this_edges) in enumerate(
        zip(text_chunk_ids, edges)
    ):
        for chunk_id in this_text_chunk_ids:
            if chunk_id in all_text_chunks_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_chunks_lookup
                    and chunk_id in all_one_hop_text_chunks_lookup[e[1]]
                ):
                    relation_counts += 1  # number of times a node's text chunk is shared by its direct neighours
            all_text_chunks_lookup[chunk_id] = {
                "data": await text_chunks_db.get_by_id(chunk_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_chunks_lookup.values()]):
        display_warning(
            "Text chunks are missing, potentially corrupted text data storage"
        )
    all_text_chunks = [
        {"id": k, **v} for k, v in all_text_chunks_lookup.items() if v is not None
    ]
    # node datas are ordered by their similarity to query, therefore index also follows this order
    # within the most similar node chunks, the chunks that are shared most frequently among neighbours will
    # get pushed to the top
    all_text_chunks = sorted(
        all_text_chunks, key=lambda x: (x["order"], -x["relation_counts"])
    )
    all_text_chunks = truncate_list_by_token_size(
        all_text_chunks,
        key=lambda x: x["data"]["content"],
        max_token_size=QueryParam.local_max_token_for_text_unit,
    )
    all_text_chunks: list[TextChunkSchema] = [t["data"] for t in all_text_chunks]
    return all_text_chunks


async def get_edge_data_from_similar_entities(
    node_datas: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
):
    # A reminder, node_datas are similar entities related to a user query ranked by
    # their similarity in descending order
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    # list of edge data
    _all_edges_data = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[
            knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges
        ]  # edge degree: sum of degrees of connected nodes
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, _all_edges_data, all_edges_degree)
        if v is not None
    ]
    # edge degree, edges between highly connected nodes get higher rank, and if tied
    # break the tie by edge weight
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=QueryParam.local_max_token_for_local_context,
    )
    return all_edges_data
