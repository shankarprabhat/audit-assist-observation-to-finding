import os
import time
import nest_asyncio
import traceback
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.core.settings import Settings
from llama_index.core.workflow import Event, Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.schema import NodeWithScore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.core.retrievers import VectorIndexRetriever
import config

from dotenv import load_dotenv
load_dotenv() # Load variables from .env into the environment

class RAGWorkflow(Workflow):
    def __init__(
        self,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        use_cloud_embedding: bool = False,
        use_chroma_db: bool = False,
        chroma_host: str = None,
        chroma_port: int = None,
        chroma_collection_name: str = "audit_findings_collection",
        chroma_api_key: str = None,
        persist_dir: str = "./storage"
    ):
        super().__init__()

        # --- Embedding Model Setup ---
        self.embed_model = None 
        if use_cloud_embedding:
            hf_embedding_api_url = os.environ.get("HF_EMBEDDING_API_URL")
            hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

            if not hf_embedding_api_url or not hf_token:
                raise ValueError(
                    "HF_EMBEDDING_API_URL and HUGGINGFACEHUB_API_TOKEN must be set "
                    "in environment variables for cloud embedding when use_cloud_embedding is True."
                )
            print(f"Using Hugging Face Inference API for embeddings: {hf_embedding_api_url}")
            self.embed_model = HuggingFaceInferenceAPIEmbedding(
                api_url=hf_embedding_api_url,
                token=hf_token,
                model_name=embedding_model
            )
        else:
            custom_cache_dir = os.path.join(os.getcwd(), "fastembed_models_cache")
            os.makedirs(custom_cache_dir, exist_ok=True)
            print(f"Using local FastEmbed for embeddings: {embedding_model}")
            self.embed_model = FastEmbedEmbedding(model_name=embedding_model, cache_dir=custom_cache_dir)

        Settings.embed_model = self.embed_model

        self.index = None
        self.use_chroma_db = use_chroma_db

        # --- Vector Store Setup (ChromaDB or Local Disk) ---
        if self.use_chroma_db:
            if not chroma_host or not chroma_port:
                raise ValueError("Chroma host and port must be provided when use_chroma_db is True.")
            
            chroma_headers = {}
            if chroma_api_key:
                chroma_headers["X-Chroma-Token"] = chroma_api_key

            # Using https=True is often necessary for cloud ChromaDB
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                headers=chroma_headers,
                # ssl=True # Uncomment if your Chroma Cloud instance requires SSL/HTTPS
            )
            try:
                # Attempt to get or create collection. This can fail if client config is bad.
                self.chroma_collection = self.chroma_client.get_or_create_collection(name=chroma_collection_name)
                self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
                print(f"Configured to use ChromaDB collection: '{self.chroma_collection.name}' at {chroma_host}:{chroma_port}")
                self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            except Exception as e:
                raise ConnectionError(f"Failed to connect or create ChromaDB collection: {e}") from e
        else:
            self.persist_dir = persist_dir
            if not os.path.exists(self.persist_dir):
                os.makedirs(self.persist_dir, exist_ok=True)
            self.storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            print(f"Configured to use local disk persistence at: {self.persist_dir}")
        # --- End Vector Store Setup ---

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest documents."""
        dirname = ev.get("dirname")
        if not dirname:
            print("No directory name provided for ingestion.")
            return None

        documents = SimpleDirectoryReader(dirname).load_data()
        if not documents:
            print(f"No documents found in directory: {dirname}. Skipping ingestion.")
            return None

        index_exists = False
        if self.use_chroma_db:
            try:
                # Check if collection has documents. This assumes collection exists.
                if self.chroma_collection.count() > 0:
                    index_exists = True
            except Exception as e:
                print(f"Warning: Could not check Chroma collection count. Assuming no index or error: {e}")
        else: # Local disk persistence
            if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
                index_exists = True

        if index_exists:
            print(f"Loading index from {'ChromaDB' if self.use_chroma_db else self.persist_dir}...")
            if self.use_chroma_db:
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=self.vector_store,
                    embed_model=self.embed_model
                )
            else: # Local disk
                self.index = load_index_from_storage(storage_context=self.storage_context)
            print("Index loaded successfully.")
        else:
            print(f"Creating new index and persisting to {'ChromaDB' if self.use_chroma_db else self.persist_dir}...")
            for doc in documents:
                print(f"Document ID: {doc.id_}, File Name: {doc.metadata.get('file_name')}, File Path: {doc.metadata.get('file_path')}")
            
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )
            # Only persist explicitly for local disk, Chroma handles persistence internally
            if not self.use_chroma_db:
                self.index.storage_context.persist(persist_dir=self.persist_dir)
            
            print(f"Index created and persisted successfully in {'ChromaDB' if self.use_chroma_db else self.persist_dir}.")

        return StopEvent(result=self.index)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point for RAG retrieval."""
        query = ev.get("query")
        index = ev.get("index") or self.index
        document_filename = ev.get("document_filename")

        filters = None
        if document_filename:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="file_name", value=document_filename, operator=FilterOperator.IN, case_sensitive=False)],
                condition=FilterCondition.AND
            )

        if not query:
            print("No query provided for retrieval.")
            return None

        if index is None:
            print("Index is empty, attempting to load from vector store for query!")
            try:
                if self.use_chroma_db:
                    self.index = VectorStoreIndex.from_vector_store(
                        vector_store=self.vector_store,
                        embed_model=self.embed_model
                    )
                else:
                    self.index = load_index_from_storage(storage_context=self.storage_context)
                index = self.index
                print("Index loaded from existing vector store for query.")
            except Exception as e:
                print(f"Could not load index from vector store: {e}")
                return None

        # Check if index is still None after attempted load
        if index is None:
            print("Failed to load index; cannot perform retrieval.")
            return None

        retriever = VectorIndexRetriever(index=index, filters=filters, similarity_top_k=3)
        nodes = await retriever.aretrieve(query)
        await ctx.set("query", query)
            
        return StopEvent(result=nodes)
        
    async def query(self, query_text: str, document_filename: str = None) -> list[NodeWithScore]:
        """Helper method to perform a RAG retrieval query, returning only relevant nodes."""
        retrieved_nodes: list[NodeWithScore]= await self.run(
            query=query_text,
            index=self.index,
            document_filename=document_filename
        )
        return retrieved_nodes


    async def ingest_documents(self, directory: str):
        """Helper method to ingest documents."""
        result = await self.run(dirname=directory)
        self.index = result
        return result

# Example usage
async def main_function(run_config):
    nest_asyncio.apply()
    
    workflow = RAGWorkflow(
        embedding_model=run_config["embedding_model"],
        use_cloud_embedding=run_config.get("use_cloud_embedding", False),
        use_chroma_db=run_config.get("use_chroma_db", False),
        chroma_host=run_config.get("chroma_host"),
        chroma_port=run_config.get("chroma_port"),
        chroma_collection_name=run_config.get("chroma_collection_name", "audit_findings_collection"),
        chroma_api_key=run_config.get("chroma_api_key"),
        persist_dir=run_config.get("persist_dir", "./storage")
    )
    
    print("\n--- Starting Ingestion ---")
    await workflow.ingest_documents("source")
    print("--- Ingestion Complete ---")

    success = True
    try:
        tic = time.time()
        
        question = run_config["question"]

        specific_document = run_config.get("source_file", None)
        
        print(f"\n--- Starting Retrieval for Query: '{question}' ---")
        retrieved_nodes = await workflow.query(question, document_filename=specific_document)
        
        print(f"\n--- Retrieved Nodes for Query: '{question}' ---")
        top_node_text = ""
        if retrieved_nodes:
            for i, node_with_score in enumerate(retrieved_nodes):
                print(f"Node {i+1}:")
                print(f"  Score: {node_with_score.score:.4f}")
                print(f"  Filename: {node_with_score.node.metadata.get('file_name', 'N/A')}")
                print(f"  Text (first 200 chars): {node_with_score.node.text[:200]}...")
                print("-" * 30)
                if i == 0:
                    top_node_text = retrieved_nodes[0].node.text
            
            unique_retrieved_filenames = list(set([node.metadata.get("file_name", "Unknown") for node in retrieved_nodes]))
            print(f"Retrieved from documents: {', '.join(unique_retrieved_filenames)}")

        else:
            print("No relevant nodes found.")

        toc = time.time()
        print(f"\nRetrieval completed in {toc - tic:.2f} seconds.")

    except Exception as e:
        print(f"Error during retrieval for question '{question}': {e}")
        traceback.print_exc()
        success = False
        top_node_text = ""
        
    print('\ntop node text: ', top_node_text)
    return top_node_text, success
    
if __name__ == "__main__":
    import asyncio

    embedding_model_choice = "BAAI/bge-large-en-v1.5"
    
    # --- GLOBAL CONFIGURATION FLAGS ---
    # Set to True to use HF Cloud Endpoint for embeddings; False for local FastEmbed
    USE_CLOUD_EMBEDDING = config.USE_CLOUD_EMBEDDING
    
    # Set to True to use ChromaDB as vector store; False for local disk persistence
    USE_CHROMA_DB = config.USE_CHROMA_DB

    # --- ChromaDB Connection Details (only used if USE_CHROMA_DB is True) ---
    # For local Chroma server (if you're running one):
    # chroma_host_config = "localhost"
    # chroma_port_config = 8000 
    # chroma_api_key_config = None 

    # For Chroma Cloud (recommended for Vercel deployment):
    # Make sure these are set in your .env file or environment variables
    chroma_host_config = os.environ.get("CHROMA_CLOUD_HOST")
    # Convert port to int, handle potential None or missing value
    chroma_port_config = int(os.environ.get("CHROMA_CLOUD_PORT")) if os.environ.get("CHROMA_CLOUD_PORT") else None
    chroma_api_key_config = os.environ.get("CHROMA_CLOUD_API_KEY")

    # --- Local Persistence Directory (only used if USE_CHROMA_DB is False) ---
    local_persist_dir = "./storage"

    run_config_for_retrieval = {
        "question": "Informed consent did not mention the expected duration of the subject's participation",
        "embedding_model": embedding_model_choice,
        "source_file": "21 CFR Part 50.pdf",
        "use_cloud_embedding": USE_CLOUD_EMBEDDING,
        "use_chroma_db": USE_CHROMA_DB,
        "chroma_host": chroma_host_config,
        "chroma_port": chroma_port_config,
        "chroma_collection_name": "audit_findings_bge_collection", # Can be any name
        "chroma_api_key": chroma_api_key_config,
        "persist_dir": local_persist_dir
    }
    
    try:
        nest_asyncio.apply()
        asyncio.run(main_function(run_config_for_retrieval))
    except Exception as e:
        print(f"An error occurred in main execution: {e}")
        traceback.print_exc()