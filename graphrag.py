import asyncio
from dataclasses import asdict, dataclass
from typing import Type, Union, cast
from src.default_storage_utils import (
    StorageNameSpace,
    BaseKVStorage,
    BaseGraphStorage,
    BaseVectorStorage,
    JsonKVStorage,
    SimpleVectorStorage,
    NetworkXStorage,
)
from src.graph_ops import (
    chunking_by_token_size,
    extract_entities,
    generate_community_report,
)
from src.rag_ops import local_query, global_query
from src.configs import ConfigParam
from src.log_utils import display_spam, display_info, display_warning
from src.misc_utils import create_folder, compute_mdhash_id


@dataclass
class GraphRAG:
    # graph mode
    enable_local: bool = True

    # storage attr
    key_string_value_json_storage_cls: Type[BaseKVStorage] = JsonKVStorage
    vector_db_storage_cls: Type[BaseVectorStorage] = SimpleVectorStorage
    graph_storage_cls: Type[BaseGraphStorage] = NetworkXStorage

    def __post_init__(self):
        _print_config = ",\n  ".join(
            [f"{k} = {v}" for k, v in asdict(ConfigParam()).items()]
        )
        display_spam(f"GraphRAG initialized with param:\n\n  {_print_config}\n")

        create_folder(ConfigParam.cache_dir)
        display_info(f"Created and set work directory as: {ConfigParam.cache_dir}")

        self.full_docs = self.key_string_value_json_storage_cls(namespace="full_docs")

    def insert(self, doc_text_string):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_insert(doc_text_string))

    def batch_insert(self, doc_text_list):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_batch_insert(doc_text_list))

    def query(self, query: str, doc_key: str, mode: str = "local"):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_query(query, doc_key, mode))

    async def async_query(self, query: str, doc_key: str, mode: str = "local"):
        (
            text_chunks,
            community_reports,
            knowledge_graph,
            entities_vdb,
        ) = self.intialize_storage_classes_per_doc(doc_key)

        if not community_reports._data:
            return "no response due to empty community reports"

        if mode == "local":
            if not entities_vdb.embeddings:
                return "no response due to empty vector datebase"
            response = await local_query(
                query=query,
                knowledge_graph_inst=knowledge_graph,
                entities_vdb=entities_vdb,
                community_reports=community_reports,
                text_chunks_db=text_chunks,
            )
        elif mode == "global":
            response = await global_query(
                query=query,
                knowledge_graph_inst=knowledge_graph,
                community_reports=community_reports,
            )
        else:
            raise ValueError(f"Unknown mode {mode}")
        return response

    async def async_batch_insert(
        self, docs: list[str], param: ConfigParam = ConfigParam()
    ):
        doc_keys = await asyncio.gather(
            *[self.async_insert(doc_text, param) for doc_text in docs]
        )
        return doc_keys

    def intialize_storage_classes_per_doc(self, doc_key: str):
        text_chunks = self.key_string_value_json_storage_cls(
            namespace=f"{doc_key}_text_chunks"
        )

        community_reports = self.key_string_value_json_storage_cls(
            namespace=f"{doc_key}_community_reports"
        )
        knowledge_graph = self.graph_storage_cls(namespace=f"{doc_key}_knowledge_graph")

        entities_vdb = (
            self.vector_db_storage_cls(
                namespace=f"{doc_key}_node_vector_db",
                meta_fields={"entity_name"},
            )
            if self.enable_local
            else None
        )
        return text_chunks, community_reports, knowledge_graph, entities_vdb

    async def async_insert(
        self, doc_text_string: str, param: ConfigParam = ConfigParam()
    ):
        # Adding a new document to JsonStorage
        new_doc = {
            compute_mdhash_id(doc_text_string.strip(), prefix="doc-"): {
                "content": doc_text_string.strip()
            }
        }
        doc_key = list(new_doc.keys())[0]
        new_doc_key = await self.full_docs.filter_keys(list(new_doc.keys()))
        new_doc_key = next(iter(new_doc_key)) if len(new_doc_key) else None
        if not new_doc_key:
            display_warning(f"Document already exists in storage, insertion skipped")
            return doc_key
        display_info(f"[New Doc] inserting doc: {new_doc_key}")

        (
            self.text_chunks,
            self.community_reports,
            self.knowledge_graph,
            self.entities_vdb,
        ) = self.intialize_storage_classes_per_doc(new_doc_key)

        # chunkfiy document into text chunks of equal token size
        # chunks follow TextChunkSchema and are added to a Json Storage object
        inserting_chunks = {
            compute_mdhash_id(dp["content"], prefix="chunk-"): {
                **dp,
                "full_doc_id": new_doc_key,
            }
            for dp in chunking_by_token_size(
                new_doc[new_doc_key]["content"],
                overlap_token_size=param.chunk_overlap_token_size,
                max_token_size=param.chunk_token_size,
            )
        }
        _add_chunk_keys = await self.text_chunks.filter_keys(
            list(inserting_chunks.keys())
        )
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }
        if not len(inserting_chunks):
            display_warning(
                f"All chunks already existed in the {new_doc_key}_text_chunks.json storage, insertion skipped"
            )
            return doc_key
        display_info(
            f"[New Chunks] inserting {len(inserting_chunks)} chunks in {new_doc_key}_text_chunks.json"
        )

        # clean all cache for community report json storage
        await self.community_reports.drop()

        display_info(
            f"[Asynchronous Entity and Relationship Extractions from {new_doc_key}]..."
        )
        self.knowledge_graph = await extract_entities(
            chunks=inserting_chunks,
            entity_vdb=self.entities_vdb,
            knwoledge_graph_inst=self.knowledge_graph,
        )
        if self.knowledge_graph is None:
            display_warning("No new entities found")
            return doc_key

        display_info(
            f"[Community discovery via leiden clustering from {new_doc_key}]..."
        )
        await self.knowledge_graph.clustering(param.graph_cluster_algorithm)
        breakpoint() # hierachical leiden clustering

        display_info(
            f"[Asynchronous community report generation using LLM for {new_doc_key}]..."
        )
        self.community_reports = await generate_community_report(
            community_report_kv=self.community_reports,
            knwoledge_graph_inst=self.knowledge_graph,
        )

        display_info(
            f"[Commit upsert of {new_doc_key} and its_chunks to storage classes]..."
        )
        await self.full_docs.upsert(new_doc)
        await self.text_chunks.upsert(inserting_chunks)

        display_info("[Save all storage classes]")
        await self._insert_done()
        breakpoint()
        return new_doc_key

    async def _insert_done(self):
        tasks = []
        for storage_inst in [
            self.full_docs,
            self.text_chunks,
            self.community_reports,
            self.entities_vdb,
            self.knowledge_graph,
        ]:
            if storage_inst is None:
                continue
            tasks.append(cast(StorageNameSpace, storage_inst).index_done_callback())
        await asyncio.gather(*tasks)
