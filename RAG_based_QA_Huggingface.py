import chromadb
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
import os

# --- Configuration ---
PDF_PERSIST_DIR = "./chroma_db_pdfs"
COLLECTION_NAME = "pdf_documents_hf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Or another suitable HF model
LLM_MODEL_NAME = "google/flan-t5-large" # Example LLM from HF for generation

# --- Initialize Components ---
client = chromadb.PersistentClient(path=PDF_PERSIST_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModel.from_pretrained(LLM_MODEL_NAME)

# --- Utility Functions ---
def generate_pdf_id(pdf_path):
    """Generates a unique ID for a PDF based on filename."""
    return os.path.basename(pdf_path)

def check_if_pdf_exists(pdf_id):
    """Checks if a PDF ID exists in the ChromaDB collection."""
    results = collection.get(
        where={"pdf_id": pdf_id},
        limit=1
    )
    return len(results["ids"]) > 0

def load_and_embed_pdf_hf(pdf_path, pdf_id):
    """Loads, chunks, embeds (using Hugging Face Transformers), and stores a PDF in ChromaDB."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    embeddings = []
    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(chunk.page_content, padding=True, truncation=True, return_tensors='pt')
            outputs = embedding_model(**inputs)
            # Mean pooling to get a fixed-size sentence embedding
            embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy()[0])

    metadatas = [{"pdf_id": pdf_id, "page": chunk.metadata["page"]} for chunk in chunks]
    ids = [f"{pdf_id}_page_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=[chunk.page_content for chunk in chunks]
    )
    print(f"PDF '{pdf_id}' embedded and stored in ChromaDB using Hugging Face Transformers.")

def embed_question_hf(question):
    """Embeds a question using Hugging Face Transformers."""
    with torch.no_grad():
        inputs = tokenizer(question, padding=True, truncation=True, return_tensors='pt')
        outputs = embedding_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()[0]

def generate_llm_answer_hf(question, context):
    """Generates an answer using a Hugging Face Transformer LLM."""
    prompt = f"context: {context}\nquestion: {question}\nanswer:"
    inputs = llm_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=200)
    answer = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def answer_question_hf(pdf_id, question):
    """Answers a question based on the content of a specific PDF in ChromaDB using HF embeddings and LLM."""
    question_embedding = embed_question_hf(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5,
        where={"pdf_id": pdf_id}
    )

    if results and results["documents"]:
        context = "\n\n".join(results["documents"][0])
        answer = generate_llm_answer_hf(question, context)
        return answer
    else:
        return "No relevant information found in the PDF."

# --- Workflow ---
def process_pdf_and_answer_hf(pdf_path, question):
    pdf_id = generate_pdf_id(pdf_path)

    if check_if_pdf_exists(pdf_id):
        print(f"PDF '{pdf_id}' already exists in ChromaDB. Proceeding to question answering.")
        answer = answer_question_hf(pdf_id, question)
        print(f"Answer: {answer}")
    else:
        print(f"PDF '{pdf_id}' is being uploaded for the first time.")
        load_and_embed_pdf_hf(pdf_path, pdf_id)
        answer = answer_question_hf(pdf_id, question)
        print(f"Answer: {answer}")

# --- Example Usage ---
if __name__ == "__main__":

    pdf_file1 = "pdf_data/FRS-04 1.Testing.pdf"
    pdf_file2 = "pdf_data/FRS-04 2.Testing.pdf.pdf"
    question1_pdf1 = "What is this document about?"
    question2_pdf2 = "What are the topics discussed in the second document?"
    question3_pdf1 = "Tell me more about the content of the first PDF."

    print("\n--- Processing PDF 1 (First Time) ---")
    process_pdf_and_answer_hf(pdf_file1, question1_pdf1)

    # print("\n--- Processing PDF 2 (First Time) ---")
    # process_pdf_and_answer_hf(pdf_file2, question2_pdf2)

    # print("\n--- Processing PDF 1 Again (Shouldn't Re-embed) ---")
    # process_pdf_and_answer_hf(pdf_file1, question3_pdf1)