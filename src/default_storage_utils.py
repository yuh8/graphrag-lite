import os
import html
import json
import asyncio
import numpy as np
import networkx as nx
from functools import wraps
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TypedDict, Union, Generic, TypeVar, cast, Any
from graspologic.utils import largest_connected_component
from graspologic.partition import hierarchical_leiden
from .llm_utils import get_embedding_response_from_genai_svc
from .prompt import GRAPH_FIELD_SEP
from .configs import ConfigParam
from .log_utils import display_info, display_warning
from .misc_utils import create_folder, write_json, load_json


SingleEntitySchema = TypedDict(
    "SingleEntitySchema",
    {"entity_name": str, "entity_type": str, "description": str, "source_id": str},
)

SingleEdgeSchema = TypedDict(
    "SingleEdgeSchema",
    {
        "src_id": str,
        "tgt_id": str,
        "weight": float,
        "description": str,
        "source_id": str,
    },
)


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

SingleCommunitySchema = TypedDict(
    "SingleCommunitySchema",
    {
        "level": int,
        "title": str,
        "edges": list[tuple[str, str]],
        "nodes": list[str],
        "chunk_ids": list[str],
        "occurrence": float,
        "sub_communities": list[str],
    },
)


class CommunitySchema(SingleCommunitySchema):
    report_string: str
    report_json: dict


T = TypeVar("T")


@dataclass
class StorageNameSpace:
    namespace: str

    async def index_done_callback(self):
        """commit the storage operations after indexing"""
        pass

    async def query_done_callback(self):
        """commit the storage operations after querying"""
        pass


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    async def all_keys(self) -> list[str]:
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """return un-exist keys"""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        raise NotImplementedError

    async def drop(self):
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    async def has_node(self, node_id: str) -> bool:
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        raise NotImplementedError

    async def clustering(self, algorithm: str):
        raise NotImplementedError

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """Return the community representation with report and nodes"""
        raise NotImplementedError


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


@dataclass
class BaseVectorStorage(StorageNameSpace):
    embedding_func: EmbeddingFunc = field(
        default_factory=lambda: get_embedding_response_from_genai_svc
    )
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value
        """
        raise NotImplementedError


def thread_lock() -> callable:
    """Asynchronous lock for preventing race conditions when updating shared object"""

    lock = asyncio.Lock()

    def final_decro(func):
        @wraps(func)
        async def lock_func(*args, **kwargs):
            async with lock:
                result = await func(*args, **kwargs)
            return result

        return lock_func

    return final_decro


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        cache_dir = ConfigParam.cache_dir
        create_folder(cache_dir)
        self._file_name = os.path.join(cache_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        display_info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self) -> None:
        write_json(self._data, self._file_name)

    async def get_by_id(self, id) -> Union[dict, None]:
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None) -> list[dict]:
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    @thread_lock()
    async def upsert(self, data: dict[str, dict]):
        new_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(new_data)
        return new_data

    async def drop(self):
        self._data = {}


@dataclass
class NetworkXStorage(BaseGraphStorage):
    @staticmethod
    def load_nx_graph(file_name) -> nx.Graph:
        if os.path.exists(file_name):
            return nx.read_graphml(file_name)
        return None

    @staticmethod
    def write_nx_graph(graph: nx.Graph, file_name):
        display_info(
            f"Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        nx.write_graphml(graph, file_name)

    @staticmethod
    def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Return the largest connected component of the graph, with nodes and edges sorted in a consistent way.

        We are only considering the largest connected knowledge graph extracted from a document
        """

        graph = graph.copy()
        graph = cast(
            nx.Graph, largest_connected_component(graph)
        )  # strict type checking
        node_mapping = {
            node: html.unescape(node.upper().strip()) for node in graph.nodes()
        }
        graph = nx.relabel_nodes(graph, node_mapping)
        return NetworkXStorage._stabilize_graph(graph)

    @staticmethod
    def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
        """Refer to https://github.com/microsoft/graphrag/index/graph/utils/stable_lcc.py
        Ensure an undirected graph with the same relationships will always be read the same way.
        """
        fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

        sorted_nodes = graph.nodes(data=True)
        # sort nodes alphabetically based on entity names
        sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

        fixed_graph.add_nodes_from(sorted_nodes)
        edges = list(graph.edges(data=True))

        # knowledge graph should be undirected, order of nodes in an edge does not matter
        if not graph.is_directed():

            def _sort_source_target(edge):
                source, target, edge_data = edge
                if source > target:
                    temp = source
                    source = target
                    target = temp
                return source, target, edge_data

            edges = [_sort_source_target(edge) for edge in edges]

        def _get_edge_key(source: Any, target: Any) -> str:
            return f"{source} -> {target}"

        edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

        fixed_graph.add_edges_from(edges)
        return fixed_graph

    def __post_init__(self):
        cache_dir = ConfigParam.cache_dir
        create_folder(cache_dir)

        self._graphml_xml_file = os.path.join(
            cache_dir, f"graph_{self.namespace}.graphml"
        )
        preloaded_graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file)
        if preloaded_graph is not None:
            display_info(
                f"Loaded graph from {self._graphml_xml_file} with {preloaded_graph.number_of_nodes()} nodes, {preloaded_graph.number_of_edges()} edges"
            )
        self._graph = preloaded_graph or nx.Graph()
        self._clustering_algorithms = {
            "leiden": self._leiden_clustering,
        }

    async def index_done_callback(self):
        # save graph after insertion or update
        NetworkXStorage.write_nx_graph(self._graph, self._graphml_xml_file)

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        return self._graph.nodes.get(node_id)

    async def node_degree(self, node_id: str) -> int:
        # degree -> number of direct neighbors
        return self._graph.degree(node_id)

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        # sum of connected nodes' degree
        return self._graph.degree(src_id) + self._graph.degree(tgt_id)

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        return self._graph.edges.get((source_node_id, target_node_id))

    async def get_node_edges(self, source_node_id: str):
        if self._graph.has_node(source_node_id):
            return list(self._graph.edges(source_node_id))
        return None

    @thread_lock()
    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        self._graph.add_node(node_id, **node_data)

    @thread_lock()
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        self._graph.add_edge(source_node_id, target_node_id, **edge_data)

    async def clustering(self, algorithm: str):
        if algorithm not in self._clustering_algorithms:
            raise ValueError(f"Clustering algorithm {algorithm} not supported")
        await self._clustering_algorithms[algorithm]()

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        """
        async method to get the high level community schema json
        """
        results = defaultdict(
            lambda: dict(
                level=None,
                title=None,
                edges=set(),
                nodes=set(),
                chunk_ids=set(),
                occurrence=0.0,
                sub_communities=[],
            )
        )
        max_num_ids = 0
        levels = defaultdict(set)
        for node_id, node_data in self._graph.nodes(
            data=True
        ):  # return all nodes with data
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = self._graph.edges(node_id)

            # the same node can belong to a different cluster on a different level
            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                results[cluster_key]["level"] = level
                results[cluster_key]["title"] = f"Cluster {cluster_key}"
                results[cluster_key]["nodes"].add(node_id)
                results[cluster_key]["edges"].update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                results[cluster_key]["chunk_ids"].update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(results[cluster_key]["chunk_ids"]))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection

            # a sub community (a set of nodes) should be fully contained in the upper level community
            for comm in this_level_comms:
                results[comm]["sub_communities"] = [
                    c
                    for c in next_level_comms
                    if results[c]["nodes"].issubset(
                        results[comm]["nodes"]
                    )  # if sub level is fully contained
                ]

        for k, v in results.items():
            v["edges"] = list(v["edges"])
            v["edges"] = [list(e) for e in v["edges"]]
            v["nodes"] = list(v["nodes"])
            v["chunk_ids"] = list(v["chunk_ids"])
            # number of doc chunks belonging to a cluster / maximum number of chunks belonging to
            # a cluster
            v["occurrence"] = len(v["chunk_ids"]) / max_num_ids
        return dict(results)

    def _cluster_data_to_subgraphs(self, cluster_data: dict[str, list[dict[str, str]]]):
        for node_id, clusters in cluster_data.items():
            self._graph.nodes[node_id]["clusters"] = json.dumps(clusters)

    async def _leiden_clustering(self):
        graph = NetworkXStorage.stable_largest_connected_component(self._graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size=ConfigParam.max_graph_cluster_size,
            random_seed=ConfigParam.graph_cluster_seed,
        )

        # dict type hinting and initialization of defaultdict
        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)  # default value of the dictionary is an empty set
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)  # freeze dictionary structure
        __levels = {k: len(v) for k, v in __levels.items()}
        display_info(f"Each level has communities: {dict(__levels)}")
        self._cluster_data_to_subgraphs(node_communities)


@dataclass
class SimpleVectorStorage(BaseVectorStorage):
    def __post_init__(self):
        cache_dir = ConfigParam.cache_dir
        create_folder(cache_dir)
        # using compressed npz as the default storage medium
        self._db_file_name = os.path.join(cache_dir, f"{self.namespace}.npz")
        self._max_batch_size = ConfigParam.embedding_batch_num
        self.md5_keys = []
        self.entity_descriptions = []
        self.entity_names = []
        self.embeddings = []
        if os.path.exists(self._db_file_name):
            _vdb = np.load(self._db_file_name)
            self.md5_keys = list(_vdb["md5_keys"])
            self.entity_descriptions = list(_vdb["entity_descriptions"])
            self.entity_names = list(_vdb["entity_names"])
            self.embeddings = list(_vdb["entity_embeddings"])

    async def upsert(self, data: dict[str, dict]):
        """
        each entry of data is a dictionary with md5 hashsed entity_name as key
        and entity_name + description as content
        """
        display_info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            display_warning(
                f"Data to be inserted into vector db: {self.namespace} is empty, insertion skipped"
            )
            return []
        list_data_tuple = [
            (md5_key, v["entity_name"], v["content"]) for md5_key, v in data.items()
        ]
        md5_keys, entity_names, entity_descriptions = list(zip(*list_data_tuple))
        batches = [
            entity_descriptions[i : i + self._max_batch_size]
            for i in range(0, len(entity_descriptions), self._max_batch_size)
        ]

        embeddings_list = await asyncio.gather(
            *[self.embedding_func(batch) for batch in batches]
        )
        embeddings = sum(embeddings_list, [])

        self.md5_keys.extend(md5_keys)
        self.entity_descriptions.extend(entity_descriptions)
        self.entity_names.extend(entity_names)
        self.embeddings.extend(embeddings)

    async def query(self, query: str, top_k=5):
        q_embed = await self.embedding_func([query])
        norm_q = np.linalg.norm(q_embed, axis=1)
        norm_embeddings = np.linalg.norm(self.embeddings, axis=1)
        dot_prod = np.array(self.embeddings) * q_embed
        dot_prod = np.sum(dot_prod, axis=1)
        cos_sim = dot_prod / (norm_q * norm_embeddings)
        distance = 1 - (cos_sim + 1) / 2
        sorted_scores = np.sort(distance)
        sorted_idx = np.argsort(distance)
        results = [
            {
                "id": self.md5_keys[r],
                "distance": sorted_scores[idx],
                "entity_name": self.entity_names[r],
                "content": self.entity_descriptions[r],
            }
            for idx, r in enumerate(sorted_idx[:top_k])
        ]
        return results

    async def index_done_callback(self):
        np.savez_compressed(
            self._db_file_name,
            md5_keys=self.md5_keys,
            entity_descriptions=self.entity_descriptions,
            entity_names=self.entity_names,
            entity_embeddings=self.embeddings,
        )
