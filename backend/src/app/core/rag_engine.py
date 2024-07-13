import logging
from datetime import datetime, timedelta
from pathlib import Path
from tabnanny import verbose
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional
from xml.dom import IndexSizeErr

import nest_asyncio
import requests
import s3fs
from cachetools import TTLCache, cached
from dns import node
from fsspec.asyn import AsyncFileSystem
from httpx import stream
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import (
    GPTListIndex,
    ServiceContext,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
    load_indices_from_storage,
)
from llama_index.core.agent import (
    AgentRunner,
    CustomSimpleAgentWorker,
    FunctionCallingAgentWorker,
)
from llama_index.core.callbacks.base import BaseCallbackHandler, CallbackManager
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.indices.query.base import BaseQueryEngine
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.schema import Document as LlamaIndexDocument
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    MetadataFilters,
    VectorStore,
)
from llama_index.embeddings.bedrock import BedrockEmbedding, Models
from llama_index.legacy import GPTKnowledgeGraphIndex
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.readers.file import PDFReader
from openai import OpenAI, chat
from PIL.ImageShow import show

from app.chat.constants import (
    DB_DOC_ID_KEY,
    NODE_PARSER_CHUNK_OVERLAP,
    NODE_PARSER_CHUNK_SIZE,
    SYSTEM_MESSAGE,
)
from app.chat.pg_vector import get_vector_store_singleton
from app.chat.qa_response_synth import get_custom_response_synth
from app.chat.tools import get_api_query_engine_tool
from app.chat.utils import build_title_for_document
from app.core.config import settings
from app.core.security.presigned_url import convert_bucket_url_to_presigned_url
from app.models.db import MessageRoleEnum, MessageStatusEnum
from app.schema import Conversation as ConversationSchema
from app.schema import Document as DocumentSchema
from app.schema import DocumentMetadataKeysEnum, eventDocumentMetadata
from app.schema import Message as MessageSchema

logger = logging.getLogger(__name__)


logger.info("Applying nested asyncio patch")
nest_asyncio.apply()

BEDROCK_TOOL_LLM_NAME = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_CHAT_LLM_NAME = "anthropic.claude-3-sonnet-20240229-v1:0"
SIMPLE_BEDROCK_CHAT_LLM_NAME = "anthropic.claude-3-sonnet-20240229-v1:0"


def get_s3_fs() -> AsyncFileSystem:
    s3 = s3fs.S3FileSystem(
        key=settings.AWS_KEY,
        secret=settings.AWS_SECRET,
        endpoint_url=settings.S3_ENDPOINT_URL,
    )
    if not (settings.RENDER or s3.exists(settings.S3_BUCKET_NAME)):
        s3.mkdir(settings.S3_BUCKET_NAME, location="ap-southeast-1")
    return s3


def fetch_and_read_document(
    document: DocumentSchema,
) -> List[LlamaIndexDocument]:
    # Super hacky approach to get this to feature complete on time.
    # TODO: Come up with better abstractions for this and the other methods in this module.
    with TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / f"{str(document.id)}.pdf"
        with open(temp_file_path, "wb") as temp_file:
            access_url = convert_bucket_url_to_presigned_url(document.url)
            with requests.get(access_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
            temp_file.seek(0)
            reader = PDFReader()
            return reader.load_data(
                temp_file_path, extra_info={DB_DOC_ID_KEY: str(document.id)}
            )


def build_description_for_document(document: DocumentSchema) -> str:
    if DocumentMetadataKeysEnum.event_DOCUMENT in document.metadata_map:
        event_metadata = eventDocumentMetadata.parse_obj(
            document.metadata_map[DocumentMetadataKeysEnum.event_DOCUMENT]
        )
        print("good metadata used")

        return f"An event {event_metadata.doc_type.value} document ({event_metadata.filename}) published by {event_metadata.department}, on {event_metadata.date_published}."
    return "A document containing useful information that the user pre-selected to discuss with the assistant."


def index_to_query_engine(
    doc_id: str, index: VectorStoreIndex, service_context: ServiceContext
) -> BaseQueryEngine:
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key=DB_DOC_ID_KEY, value=doc_id)]
    )

    kwargs = {
        "similarity_top_k": 3,
        "filters": filters,
        "service_context": service_context,
    }

    return index.as_query_engine(**kwargs)


def index_to_query_engine_single(
    doc_ids: str, index: VectorStoreIndex, service_context: ServiceContext
) -> BaseQueryEngine:
    # filters = MetadataFilters(
    #     filters=[
    #         ExactMatchFilter(key=DB_DOC_ID_KEY, value=doc_id) for doc_id in doc_ids
    #     ]
    # )

    kwargs = {
        "similarity_top_k": 5,
        # "filters": filters,
        "service_context": service_context,
    }

    return index.as_query_engine(**kwargs)


def index_to_chat_engine(doc_id: str, index: VectorStoreIndex, llm) -> BaseQueryEngine:
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key=DB_DOC_ID_KEY, value=doc_id)]
    )
    kwargs = {"similarity_top_k": 3, "filters": filters}
    return index.as_chat_engine(llm=llm, **kwargs)


@cached(
    TTLCache(maxsize=10, ttl=timedelta(minutes=5).total_seconds()),
    key=lambda *args, **kwargs: "global_storage_context",
)
def get_storage_context(
    persist_dir: str, vector_store: VectorStore, fs: Optional[AsyncFileSystem] = None
) -> StorageContext:
    logger.info("Creating new storage context.")
    return StorageContext.from_defaults(
        persist_dir=persist_dir, vector_store=vector_store, fs=fs
    )


async def build_doc_id_to_index_map(
    service_context: ServiceContext,
    documents: List[DocumentSchema],
    fs: Optional[AsyncFileSystem] = None,
) -> Dict[str, VectorStoreIndex]:
    persist_dir = f"{settings.S3_BUCKET_NAME}"

    vector_store = await get_vector_store_singleton()

    try:
        try:
            storage_context = get_storage_context(persist_dir, vector_store, fs=fs)
        except FileNotFoundError:
            logger.info(
                "Could not find storage context in S3. Creating new storage context."
            )
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, fs=fs
            )
            storage_context.persist(persist_dir=persist_dir, fs=fs)
        index_ids = [str(doc.id) for doc in documents]
        indices = load_indices_from_storage(
            storage_context,
            index_ids=index_ids,
            service_context=service_context,
        )
        doc_id_to_index = dict(zip(index_ids, indices))
        logger.debug("Loaded indices from storage.")
    except ValueError:
        logger.error(
            "Failed to load indices from storage. Creating new indices. "
            "If you're running the seed_db script, this is normal and expected."
        )
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir, vector_store=vector_store, fs=fs
        )
        doc_id_to_index = {}
        for doc in documents:
            llama_index_docs = fetch_and_read_document(doc)
            storage_context.docstore.add_documents(llama_index_docs)
            index = VectorStoreIndex.from_documents(
                llama_index_docs,
                storage_context=storage_context,
                service_context=service_context,
            )
            index.set_index_id(str(doc.id))
            index.storage_context.persist(persist_dir=persist_dir, fs=fs)
            doc_id_to_index[str(doc.id)] = index
    return doc_id_to_index


# this is for dumb rag
async def build_single_index(
    service_context: ServiceContext,
    documents: List[DocumentSchema],
    fs: Optional[AsyncFileSystem] = None,
    force: bool = False,
):
    fullstore_filename_prefix = "fullstore"
    vector_store = await get_vector_store_singleton()
    persist_dir = f"{settings.S3_BUCKET_NAME}"
    try:
        try:
            storage_context = get_storage_context(persist_dir, vector_store, fs=fs)
        except FileNotFoundError:
            logger.info(
                "Could not find storage context in S3. Creating new storage context."
            )
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, fs=fs
            )
            storage_context.persist(
                persist_dir=persist_dir,
                fs=fs,
                vector_store_fname=fullstore_filename_prefix + "vectors.json",
                docstore_fname=fullstore_filename_prefix + "docs.json",
                graph_store_fname=fullstore_filename_prefix + "graph.json",
                index_store_fname=fullstore_filename_prefix + "index.json",
            )
        if force:
            raise ValueError
        doc_ids = [str(doc.id) for doc in documents]
        index = load_index_from_storage(
            storage_context,
            index_id="fullstore",
            service_context=service_context,
        )
        # TODO make it so that instead of rebuilding the enitre vector store it just inserts new ones

        # for id in doc_ids:
        #     if id not in index.ref_doc_info:
        #         raise ValueError

        logger.debug("Loaded indices from storage.")
    except ValueError:
        logger.error("failed to find persisted vector store")
        storage_context = StorageContext.from_defaults(
            persist_dir=persist_dir, vector_store=vector_store, fs=fs
        )
        llama_index_docs = [fetch_and_read_document(doc) for doc in documents]
        llama_index_docs = [x for y in llama_index_docs for x in y]
        storage_context.docstore.add_documents(llama_index_docs)
        index = VectorStoreIndex.from_documents(
            llama_index_docs,
            storage_context=storage_context,
            service_context=service_context,
            show_progress=True,
        )
        index.set_index_id("fullstore")
        index.storage_context.persist(
            persist_dir=persist_dir,
            fs=fs,
            # vector_store_fname=fullstore_filename_prefix + "vectors.json",
            # docstore_fname=fullstore_filename_prefix + "docs.json",
            # graph_store_fname=fullstore_filename_prefix + "graph.json",
            # index_store_fname=fullstore_filename_prefix + "index.json",
        )
    return index


async def rebuild_vector_db(
    service_context: ServiceContext,
    documents: List[DocumentSchema],
    fs: Optional[AsyncFileSystem] = None,
):
    persist_dir = f"{settings.S3_BUCKET_NAME}"

    vector_store = await get_vector_store_singleton()
    logger.error(
        "Failed to load indices from storage. Creating new indices. "
        "If you're running the seed_db script, this is normal and expected."
    )
    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir, vector_store=vector_store, fs=fs
    )
    doc_id_to_index = {}
    for doc in documents:
        llama_index_docs = fetch_and_read_document(doc)
        storage_context.docstore.add_documents(llama_index_docs)
        index = VectorStoreIndex.from_documents(
            llama_index_docs,
            storage_context=storage_context,
            service_context=service_context,
        )
        index.set_index_id(str(doc.id))
        index.storage_context.persist(persist_dir=persist_dir, fs=fs)
        doc_id_to_index[str(doc.id)] = index


def get_chat_history(
    chat_messages: List[MessageSchema],
) -> List[ChatMessage]:
    """
    Given a list of chat messages, return a list of ChatMessage instances.

    Failed chat messages are filtered out and then the remaining ones are
    sorted by created_at.
    """
    # pre-process chat messages
    chat_messages = [
        m
        for m in chat_messages
        if m.content.strip() and m.status == MessageStatusEnum.SUCCESS
    ]
    # TODO: could be a source of high CPU utilization
    # chat_messages = sorted(chat_messages, key=lambda m: m.created_at)

    chat_history = []
    for message in chat_messages:
        role = (
            MessageRole.ASSISTANT
            if message.role == MessageRoleEnum.assistant
            else MessageRole.USER
        )
        chat_history.append(ChatMessage(content=message.content, role=role))

    return chat_history


def get_tool_service_context(
    callback_handlers: List[BaseCallbackHandler],
) -> ServiceContext:
    callback_manager = CallbackManager(callback_handlers)

    llm = BedrockConverse(
        temperature=0,
        model=BEDROCK_TOOL_LLM_NAME,
        aws_access_key_id=settings.AWS_KEY,
        aws_secret_access_key=settings.AWS_SECRET,
        region_name="ap-southeast-1",
        callback_manager=callback_manager,
    )
    embedding_model = BedrockEmbedding(
        model_name=Models.COHERE_EMBED_ENGLISH_V3,
        aws_access_key_id=settings.AWS_KEY,
        aws_secret_access_key=settings.AWS_SECRET,
        region_name=settings.BEDROCK_REGION,
        # TODO add access to the Bedrock API key
    )
    # Use a smaller chunk size to retrieve more granular results
    node_parser = SentenceSplitter.from_defaults(
        chunk_size=NODE_PARSER_CHUNK_SIZE,
        chunk_overlap=NODE_PARSER_CHUNK_OVERLAP,
        callback_manager=callback_manager,
    )
    service_context = ServiceContext.from_defaults(
        callback_manager=callback_manager,
        llm=llm,
        embed_model=embedding_model,
        node_parser=node_parser,
    )
    return service_context


async def get_chat_engine(
    callback_handler: BaseCallbackHandler,
    conversation: ConversationSchema,
) -> AgentRunner:
    service_context = get_tool_service_context([callback_handler])
    s3_fs = get_s3_fs()
    doc_id_to_index = await build_doc_id_to_index_map(
        service_context, conversation.documents, fs=s3_fs
    )
    for key, val in doc_id_to_index.items():
        doc_id_to_index[key]._callback_manager = service_context.callback_manager

    id_to_doc: Dict[str, DocumentSchema] = {
        str(doc.id): doc for doc in conversation.documents
    }

    vector_query_engine_tools = [
        QueryEngineTool(
            query_engine=index_to_query_engine(
                doc_id, index, service_context=service_context
            ),
            metadata=ToolMetadata(
                name=eventDocumentMetadata.parse_obj(
                    id_to_doc[doc_id].metadata_map[
                        DocumentMetadataKeysEnum.event_DOCUMENT
                    ]
                ).filename,
                description=build_description_for_document(id_to_doc[doc_id]),
            ),
        )
        for doc_id, index in doc_id_to_index.items()
    ]
    response_synth = get_custom_response_synth(service_context, conversation.documents)
    qualitative_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=vector_query_engine_tools,
        service_context=service_context,
        response_synthesizer=response_synth,
        verbose=settings.VERBOSE,
        use_async=True,
    )
    # SubQuestionQueryEngine()

    top_level_sub_tools = [
        QueryEngineTool(
            query_engine=qualitative_question_engine,
            metadata=ToolMetadata(
                name="qualitative_question_engine",
                description="""
user can ask questions about the events they have selected""",
            ),
        ),
    ]

    chat_llm = BedrockConverse(
        temperature=0,
        model=BEDROCK_TOOL_LLM_NAME,
        aws_access_key_id=settings.AWS_KEY,
        aws_secret_access_key=settings.AWS_SECRET,
        region_name="us-east-1",
    )
    chat_messages: List[MessageSchema] = conversation.messages
    chat_history = get_chat_history(chat_messages)
    logger.debug("Chat history: %s", chat_history)

    if conversation.documents:
        doc_titles = "\n".join(
            "- " + build_title_for_document(doc) for doc in conversation.documents
        )
    else:
        doc_titles = "No documents selected."

    curr_date = datetime.utcnow().strftime("%Y-%m-%d")

    if not chat_history or chat_history[0].role != MessageRole.SYSTEM:
        chat_history = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=SYSTEM_MESSAGE.format(
                    doc_titles=doc_titles, curr_date=curr_date
                ),
            )
        ] + chat_history

    chat_engine = FunctionCallingAgentWorker.from_tools(
        tools=top_level_sub_tools,
        llm=chat_llm,
        prefix_messages=chat_history,
        verbose=settings.VERBOSE,
        # system_prompt=SYSTEM_MESSAGE.format(doc_titles=doc_titles, curr_date=curr_date),
        callback_manager=service_context.callback_manager,
        max_function_calls=3,
    ).as_agent()

    return chat_engine


# OpenAIAgent


async def get_chat_engine_simplest(
    callback_handler: BaseCallbackHandler,
    conversation: ConversationSchema,
):
    service_context = get_tool_service_context([callback_handler])
    s3_fs = get_s3_fs()
    index = await build_single_index(service_context, conversation.documents, fs=s3_fs)
    doc_ids = [doc.id for doc in conversation.documents]
    id_to_doc: Dict[str, DocumentSchema] = {
        str(doc.id): doc for doc in conversation.documents
    }

    chat_llm = BedrockConverse(
        temperature=0,
        model=SIMPLE_BEDROCK_CHAT_LLM_NAME,
        aws_access_key_id=settings.AWS_KEY,
        aws_secret_access_key=settings.AWS_SECRET,
        region_name="ap-southeast-1",
    )
    chat_messages: List[MessageSchema] = conversation.messages
    chat_history = get_chat_history(chat_messages)
    logger.debug("Chat history: %s", chat_history)

    if conversation.documents:
        doc_titles = "\n".join(
            "- " + build_title_for_document(doc) for doc in conversation.documents
        )
    else:
        doc_titles = "No documents selected."

    curr_date = datetime.utcnow().strftime("%Y-%m-%d")

    if not chat_history or chat_history[0].role != MessageRole.SYSTEM:
        chat_history = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=SYSTEM_MESSAGE.format(
                    doc_titles=doc_titles, curr_date=curr_date
                ),
            )
        ] + chat_history
    kwargs = {
        "similarity_top_k": 3,
    }
    chat_engine = index.as_chat_engine(
        # llm=chat_llm,
        prefix_messages=chat_history,
        chat_mode=ChatMode.CONTEXT,
        # callback_manager=service_context.callback_manager,
        verbose=True,
        service_context=service_context,
        stream=True,
        **kwargs,
    )

    return chat_engine
