import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import os

# Load a pre-trained cross-encoder model
reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize ChromaDB client (persistent or in-memory)
client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "pdf_documents_med"

# # delete the old collection
# if collection_name in client.list_collections():
    # client.delete_collection(name=collection_name)

collection = client.get_or_create_collection(name=collection_name,metadata={"hnsw:space": "cosine"})  # cosine, l2

# Initialize embedding model
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# embedding_model = SentenceTransformer('NeuML/pubmedbert-base-embeddings')
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") # Or your chosen embedding model
embedding_model = SentenceTransformer("all-mpnet-base-v2") # Or your chosen embedding model
# Or to use S-PubMedBert-MS-MARCO:
# embedding_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# Or to use a MedEmbed model:
# embedding_model = SentenceTransformer('abhinand/MedEmbed-base-v0.1')

# PDF_PERSIST_DIR = "./chroma_db_pdfs"
# COLLECTION_NAME = "pdf_documents_hf"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Or another suitable HF model
# LLM_MODEL_NAME = "google/flan-t5-large" # Example LLM from HF for generation

# --- Initialize Components ---
# client = chromadb.PersistentClient(path=PDF_PERSIST_DIR)
# collection = client.get_or_create_collection(name=COLLECTION_NAME)

# tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
# embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

# llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
# llm_model = AutoModel.from_pretrained(LLM_MODEL_NAME)

class Generate_LLM_Reponse:

    def generate_llm_answer_hf_pipeline(self, question,context):
        # from langchain.llms import HuggingFacePipeline
        # from langchain_community.llms import HuggingFacePipeline
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
        print('\nquestion: ', question)
        print('context: ', context)
        print('\n')

        # Load the tokenizer and model directly (similar to the direct Transformers example)
        model_name = "google/flan-t5-large"
        model_name = "google/flan-t5-base"
        llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Create a Hugging Face pipeline for text generation
        pipeline = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_length=200)

        # pipeline = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer)
        # pipeline = pipeline("summarization", model=llm_model, tokenizer=llm_tokenizer)

        # This one doesnt work well
        # pipeline = pipeline("question-answering", model=llm_model, tokenizer=llm_tokenizer)

        # Initialize Langchain's HuggingFacePipeline LLM
        llm = HuggingFacePipeline(pipeline=pipeline)

        """Generates an answer using a Langchain HuggingFacePipeline LLM."""
        prompt = f"context: {context}\nquestion: {question}\nanswer:"
        
        # create a prompt with example
        # example_question = "How many enzymes are encoded by HIV Pol gene"
        # example_answer = "The HIV Pol gene encodes three enzymes (with four enzymatic activities): protease, reverse transcriptase and integrase."
        
        # example_question = "Name a Project Leader in this report"
        # example_answer = "One of the project leader is Named Chen Xiaoyan"
        
        # prompt = f"context: {context}\n example_question: {example_question} \nexample_answer: {example_answer}\nquestion: {question}\nanswer:"
        prompt = f"context: {context}\n question: {question}\n Answer the question based on the document of the text:"
        
        answer = llm.invoke(prompt)
        # print('answer before strip: ', answer)
        return answer.strip()

    def generate_llm_answer_local_download(self, question, context):

        from langchain.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

        # Specify the path to your locally downloaded model
        local_model_path = "/path/to/your/local/flan-t5-large"  # Replace with the actual path

        # Load the tokenizer and model from the local directory
        try:
            llm_tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            llm_model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
        except Exception as e:
            print(f"Error loading local model: {e}")
            print("Make sure the path '{local_model_path}' is correct and contains the model files.")
            exit()

        # Create a Hugging Face pipeline for text generation
        pipeline = pipeline("text2text-generation", model=llm_model, tokenizer=llm_tokenizer, max_length=200)

        # Initialize Langchain's HuggingFacePipeline LLM
        llm = HuggingFacePipeline(pipeline=pipeline)

        """Generates an answer using a Langchain HuggingFacePipeline with a local HF model."""
        prompt = f"context: {context}\nquestion: {question}\nanswer:"
        answer = llm(prompt)
        return answer.strip()

    def generate_llm_answer_hf_hub(self, question, context):

        # from langchain_community.llms import HuggingFaceEndpoint      
        from langchain_huggingface import HuggingFaceEndpoint

        HF_TOKEN = os.getenv("HF_TOKEN")
        # os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
        # HUGGINGFACEHUB_API_TOKEN = HF_TOKEN
        

        """Generates an answer using a Langchain HuggingFaceHub LLM."""
        # llm = HuggingFaceEndpoint(repo_id="google/flan-t5-large", huggingfacehub_api_token=HF_TOKEN) # model_kwargs={"temperature": 0.5, "max_length": 200},
        # prompt = f"context: {context}\nquestion: {question}\nanswer:"
        # answer = llm.invoke(prompt)

        from langchain.chains import LLMChain
        from langchain_core.prompts import PromptTemplate

        repo_id="google/flan-t5-large"
        # repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

        template = """Question: {question} Answer: Let's think step by step."""

        prompt = PromptTemplate.from_template(template)

        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            max_length=128,
            temperature=0.5,
            huggingfacehub_api_token=HF_TOKEN,
            task="text2text-generation"
        )
        llm_chain = prompt | llm
        answer = llm_chain.invoke({"question": question})

        return answer.strip()


def generate_pdf_id(pdf_path):
    """Generates a unique ID for a PDF (e.g., based on filename)."""
    import os
    return os.path.basename(pdf_path)

def check_if_pdf_exists(pdf_id):
    """Checks if a PDF ID exists in the ChromaDB collection.""" 
    results = collection.get(
        where={"pdf_id": pdf_id},
        limit=1  # We only need to find one
    )
    # print('results for document search: ', results)
    return len(results["ids"]) > 0

def load_and_embed_pdf(pdf_path, pdf_id):
    """Loads, chunks, embeds, and stores a PDF in ChromaDB."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    all_text_as_string = "\n\n".join(chunk.page_content for chunk in chunks)
    output_file = "combined_pdf_text.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(all_text_as_string)

    embeddings = embedding_model.encode([chunk.page_content for chunk in chunks])
    metadatas = [{"pdf_id": pdf_id, "page": chunk.metadata["page"]} for chunk in chunks]
    ids = [f"{pdf_id}_page_{i}" for i in range(len(chunks))] # Unique IDs for chunks

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=[chunk.page_content for chunk in chunks]
    )
    print(f"PDF '{pdf_id}' embedded and stored in ChromaDB.")
    return chunks

def rerank_documents(question: str, documents: list[str], reranker: CrossEncoder, top_n: int = 2):
    """
    Reranks a list of documents based on their relevance to a question using a cross-encoder.

    Args:
        question: The user's question.
        documents: A list of retrieved document strings.
        reranker: A CrossEncoder model from Sentence Transformers.
        top_n: The number of top-ranked documents to return.

    Returns:
        A list of the top_n most relevant documents (strings), ordered by relevance.
    """
    if not documents:
        return []

    # Create pairs of (question, document)
    pairs = [(question, doc) for doc in documents]

    # Get the relevance scores from the cross-encoder
    rerank_scores = reranker.predict(pairs)

    # Sort documents by their relevance score in descending order
    ranked_documents = sorted(zip(documents, rerank_scores), key=lambda x: x[1], reverse=True)

    # Return the top_n documents
    return [doc for doc, score in ranked_documents[:top_n]]


def answer_question(pdf_id, question, desired_way, chunks=None):
    """Answers a question based on the content of a specific PDF in ChromaDB."""
    question_embedding = embedding_model.encode([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,  # Number of relevant chunks to retrieve
        where={"pdf_id": pdf_id}  # Filter by the specific PDF
        # distance="l2" # Uncomment to try L2 distance
    )    

    reranking = True
    if reranking is True:
        if results: # and results['documents']:
            retrieved_documents = results['documents'][0]
            # print("Retrieved Documents:", retrieved_documents)s

            # Rerank the retrieved documents
            top_ranked_context = rerank_documents(question, retrieved_documents, reranker_model, top_n=2)

            print("Top Ranked Context Chunks:")
            for chunk in top_ranked_context:
                print(chunk[:100] + "...")
    
    
    hybrid_search = False
    if hybrid_search is True:    
        from langchain_community.vectorstores import Chroma
        from langchain.retrievers import EnsembleRetriever
        from langchain_community.retrievers import BM25Retriever
        from langchain.schema import Document
        from sentence_transformers import SentenceTransformer
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        
        # Assuming 'chunks' is your list of Langchain Document objects
        # and 'embedding_model' is your SentenceTransformer model
        
        # 1. Initialize Langchain's SentenceTransformerEmbeddings wrapper
        embedding_function = SentenceTransformerEmbeddings(model_name='NeuML/pubmedbert-base-embeddings')
        
        # 1. Convert ChromaDB to a Langchain VectorStore (if you haven't already)
        vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
        
        # 2. Initialize BM25Retriever with your document content
        bm25_retriever = BM25Retriever.from_documents(
            [Document(page_content=chunk.page_content) for chunk in chunks], k=3
        )
        
        # 3. Initialize the Chroma vector store retriever
        chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 4. Initialize the EnsembleRetriever, combining BM25 and Chroma
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever], weights=[0.5, 0.5] # Adjust weights as needed
        )
        
        # Now, when you want to retrieve context:
        # question = "What is the clinical approval number mentioned in the text?"
        relevant_docs = ensemble_retriever.get_relevant_documents(question)
        
        print("Retrieved relevant documents (hybrid search):")
        # for doc in relevant_docs:
        #     print(doc.page_content)
        
        # You would then pass 'relevant_docs' to your LLM for answer generation
        return relevant_docs
    
    elif not hybrid_search:

        if results and results["documents"]:
            context = "\n\n".join(results["documents"][0])
            # Call your LLM here with the question and context
            gr = Generate_LLM_Reponse()

            if desired_way == 'pipeline':
                answer = gr.generate_llm_answer_hf_pipeline(question, context)
            elif desired_way == 'huggingface-hub':
                answer = gr.generate_llm_answer_hf_hub(question, context)
            elif desired_way == 'local_download':
                answer = gr.generate_llm_answer_local_download(question, context)
            else:
                raise ValueError(f"Invalid desired_way: {desired_way}. Choose from 'pipeline', 'huggingface-hub', or 'local_download'.")
            return answer
        else:
            return "No relevant information found in the PDF."

# --- Workflow ---

def process_pdf_and_answer(pdf_path, question,desired_way):
    pdf_id = generate_pdf_id(pdf_path)

    pdf_exists = check_if_pdf_exists(pdf_id) # True if the PDF is already in the database

    if pdf_exists:
        print(f"PDF '{pdf_id}' already exists in ChromaDB. Proceeding to question answering.")
        answer = answer_question(pdf_id, question, desired_way)
        # print(f"\nAnswer: {answer}")
    else:
        print(f"PDF '{pdf_id}' is being uploaded for the first time.")
        import time
        tic = time.time()
        chunks = load_and_embed_pdf(pdf_path, pdf_id)
        toc = time.time()    
        print(f"time: {toc-tic}")

        answer = answer_question(pdf_id, question, desired_way, chunks)
        # print(f"\nAnswer: {answer}")

    return answer

if __name__ == "__main__":


    llm_call_ways = ['huggingface-hub','pipeline','local_download']
    desired_way = 'pipeline'
    # desired_way = 'huggingface-hub'

    # Example Usage:
    pdf_file1 = "pdf_data/ich-gcp-r2-step-5.pdf"
    pdf_file2 = "pdf_data/ich-gcp-r3-step-5.pdf"
    # question1_pdf1 = "What is the main topic?"
    question1_pdf1 = "How is quality control defined in the text?"
    question1_pdf1 = "What is good clinical practicse as defined in the text?"
    question1_pdf1 = "What is adverse drug reaction as defined in the text?"
    question1_pdf1 = "What is audit certificate as defined in the text?"
    question1_pdf1 = "What is audit report as defined in the text?"
    question1_pdf1 = "What is confidentiality as defined in the text?"
    # question1_pdf1 = "What is Coordinating committee as defined in the text?"
    
    # question1_pdf1 = "Antiretroviral drugs act on how many intraviral enzymes?"
    # question1_pdf1 = "According to their mechanism of action what are the different anti-HIV drugs? List all of them.  Think about your answer. Provide your reasoning as well."
    # question1_pdf1 = "List the different types of anti-HIV drugs mentioned in the text. Think about your answer. Provide your reasoning as well."
    # question1_pdf1 = "According to the text, what are the different anti-HIV drugs based on their mechanism of action?"
    # question1_pdf1 = "How many anti-HIV drug classes are mentioned in the text, and what are they?"

    # question1_pdf1 = "What happened after 10 days of continuous administration"
    # question1_pdf1 = "What are the results of pharmacokinetic studies of EFV?"

    # question2_pdf2 = "Who are the key people mentioned?"
    # question3_pdf1 = "Answer another question about the first PDF."
    # question1_pdf1 =  "What is the clinical approval number mentioned in the text?"
    # question1_pdf1 =  "Who is the project leader?"
    # question1_pdf1 =  "Find all the Project leaders in the given text. One of the is Zhang Ting, find others."

    # First time processing PDF 1
    answer = process_pdf_and_answer(pdf_file1, question1_pdf1,desired_way)
    print(f"\nAnswer: {answer}")

    # First time processing PDF 2
    # process_pdf_and_answer(pdf_file2, question2_pdf2)

    # Subsequent question about PDF 1 (shouldn't re-embed)
    # process_pdf_and_answer(pdf_file1, question3_pdf1)