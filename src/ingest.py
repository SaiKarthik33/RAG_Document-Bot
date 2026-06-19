import os
from pypdf import PdfReader
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

# Load API key securely
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- FINAL FIX: Added the name() method to satisfy ChromaDB ---
class SafeGeminiEmbeddingFunction:
    def name(self):
        return "SafeGeminiEmbeddingFunction"
        
    def __call__(self, input: list[str]) -> list[list[float]]:
        genai.configure(api_key=api_key)
        embeddings = []
        for text in tqdm(input, desc="Generating Embeddings", unit="chunk"):
            res = genai.embed_content(model="models/gemini-embedding-001", content=text)
            embeddings.append(res['embedding'])
        return embeddings

def extract_pdf_pages(file_path: str) -> list[dict]:
    """Extracts text page-by-page from a PDF, tracking metadata."""
    extracted_data = []
    file_name = os.path.basename(file_path)
    try:
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                clean_text = " ".join(text.split())
                extracted_data.append({
                    "text": clean_text,
                    "metadata": {"source": file_name, "page": index + 1}
                })
    except Exception as e:
        print(f"Error reading PDF {file_name}: {e}")
    return extracted_data

def chunk_extracted_pages(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """Splits page-level documents into smaller, overlapping chunks."""
    chunks = []
    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"{start}-{end}"
                }
            })
            start += (chunk_size - chunk_overlap)
    return chunks

def save_to_vector_db(chunks: list[dict], db_path: str = "./db"):
    """Embeds text chunks and saves them into a persistent disk-based ChromaDB."""
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = SafeGeminiEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name="document_knowledge_base",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    ids = [f"id_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"\nSuccessfully indexed {len(chunks)} chunks in the vector database.")

if __name__ == "__main__":
    data_dir = "./data"
    all_pages = []
    
    print("Starting document extraction...")
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(data_dir, filename)
            print(f"Reading: {filename}")
            pages = extract_pdf_pages(file_path)
            all_pages.extend(pages)
            
    if not all_pages:
        print("No PDF data found to process.")
    else:
        print(f"Extracted {len(all_pages)} total pages. Chunking text...")
        chunks = chunk_extracted_pages(all_pages)
        
        print(f"Embedding {len(chunks)} chunks and saving to database...")
        save_to_vector_db(chunks)
        print("Ingestion complete!")