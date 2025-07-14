import os
import time
import nest_asyncio
import traceback
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.settings import Settings
from llama_index.core.workflow import Event, Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.schema import NodeWithScore
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator, FilterCondition
from llama_index.core.retrievers import VectorIndexRetriever

# We will remove RetrieverEvent if it's only used for internal transition and not the final stop
# class RetrieverEvent(Event):
#     """Result of running retrieval"""
#     nodes: list[NodeWithScore]

# *** REMOVED ***: No longer need a custom RetrievalStopEvent.
# The standard StopEvent can carry the result.
# class RetrievalStopEvent(StopEvent):
#     """Event to signal the end of retrieval, carrying the nodes."""
#     result: list[NodeWithScore] # Change type to list of NodeWithScore

class RAGWorkflow(Workflow):
    def __init__(self, embedding_model="BAAI/bge-large-en-v1.5", persist_dir="./storage"):
        super().__init__()

        # Configure a custom cache directory for fastembed (recommended)
        custom_cache_dir = os.path.join(os.getcwd(), "fastembed_models_cache")
        os.makedirs(custom_cache_dir, exist_ok=True)
        self.embed_model = FastEmbedEmbedding(model_name=embedding_model, cache_dir=custom_cache_dir)

        # Configure global settings
        Settings.embed_model = self.embed_model

        self.index = None
        self.persist_dir = persist_dir

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest documents from a directory."""
        dirname = ev.get("dirname")
        if not dirname:
            return None

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"Loading index from {self.persist_dir}...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context=storage_context)
            print("Index loaded successfully.")
        else:
            print(f"Creating new index and persisting to {self.persist_dir}...")
            documents = SimpleDirectoryReader(dirname).load_data()
            for doc in documents:
                print(f"Document ID: {doc.id_}, File Name: {doc.metadata.get('file_name')}, File Path: {doc.metadata.get('file_path')}")
            self.index = VectorStoreIndex.from_documents(documents=documents)
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print("Index created and persisted successfully.")

        # This `StopEvent` is fine because it's for the 'ingest' path,
        # which is distinct from the 'retrieve' path.
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
            return None

        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = VectorIndexRetriever(index=index, filters=filters,similarity_top_k=3)
        nodes = await retriever.aretrieve(query)
        await ctx.set("query", query)
        
        # *** MODIFIED ***: Return standard StopEvent with the nodes as its result
        return StopEvent(result=nodes)
    
    # *** MODIFIED RETURN TYPE HINT ***
    async def query(self, query_text: str, document_filename: str = None) -> list[NodeWithScore]:
        """Helper method to perform a RAG retrieval query, returning only relevant nodes."""
        if self.index is None:
            raise ValueError("No documents have been ingested. Call ingest_documents first.")

        # The self.run method will return a standard StopEvent.
        # We need to access its 'result' attribute to get the nodes.
        retrieved_nodes: list[NodeWithScore]= await self.run(
            query=query_text,
            index=self.index,
            document_filename=document_filename
        )
        # Access the result attribute from the StopEvent
        return retrieved_nodes


    async def ingest_documents(self, directory: str):
        """Helper method to ingest documents."""
        # This path also uses self.run, which will lead to the 'ingest' step
        # and its StopEvent. The result here is the index itself.
        result = await self.run(dirname=directory)
        self.index = result
        return result

# Example usage
async def main_function(run_config):
    nest_asyncio.apply() # Apply nest_asyncio for Jupyter/asyncio compatibility if running outside a clean script
    
    # Initialize the workflow
    workflow = RAGWorkflow(embedding_model=run_config["embedding_model"], persist_dir="./storage")
    
    # Ingest documents
    await workflow.ingest_documents("source")
    success = True
    try:
        tic = time.time()
        
        question = run_config["question"]

        specific_document = None
        # You can uncomment this line if you want to filter by a specific document
        specific_document = run_config.get("source_file", None) # This expects a single filename string, not a list
        
        # Call the modified query method
        retrieved_nodes = await workflow.query(question, document_filename=specific_document)
        
        # Now, 'retrieved_nodes' directly contains the list of NodeWithScore objects
        # You can process them here as needed.
        
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
            
            # Extract unique filenames from retrieved nodes
            unique_retrieved_filenames = list(set([node.metadata.get("file_name", "Unknown") for node in retrieved_nodes]))
            print(f"Retrieved from documents: {', '.join(unique_retrieved_filenames)}")

        else:
            print("No relevant nodes found.")

        toc = time.time()
        print(f"\nRetrieval completed in {toc - tic:.2f} seconds.")

    except Exception as e:
        print(f"Error during retrieval for question '{question}': {e}")
        # import traceback # Ensure traceback is imported here if not global
        traceback.print_exc() # Print full traceback for debugging
        success = False
        top_node_text = ""
    
    print('\ntop node text: ', top_node_text)
    return top_node_text, success
    
    

if __name__ == "__main__":
    import asyncio
    # import traceback # Import traceback for better error reporting

    # Ensure your 'source' directory exists and contains some documents for ingestion
    # For testing, you might create a dummy 'source' directory with a .txt file inside.
    
    embedding_model_choice = "BAAI/bge-large-en-v1.5"
    
    run_config_for_retrieval = {
        "question": "Informed consent did not mention the expected duration of the subject's participation",
        "embedding_model": embedding_model_choice,
        "source_file": "21 CFR Part 50.pdf" # Make sure this matches an actual filename if used for filtering
    }
    
    # IMPORTANT: The 'source_file' in run_config_for_retrieval is passed as `specific_document`
    # to workflow.query. It needs to be the actual filename from your source folder,
    # e.g., "21 CFR Part 50.pdf", not just "21 CFR Part 50".
    
    try:
        nest_asyncio.apply()
        asyncio.run(main(run_config_for_retrieval))
    except:
        asyncio.create_task(main_function(run_config_for_retrieval))