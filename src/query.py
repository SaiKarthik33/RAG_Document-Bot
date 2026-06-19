import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class SafeGeminiEmbeddingFunction:
    def name(self):
        return "SafeGeminiEmbeddingFunction"
        
    # ChromaDB demands EXACTLY this signature: (self, input)
    def __call__(self, input: list[str]) -> list[list[float]]:
        genai.configure(api_key=api_key)
        
        # Safety catch in case a single string is passed
        if isinstance(input, str):
            input = [input]
            
        embeddings = []
        for text in input:
            res = genai.embed_content(model="models/gemini-embedding-001", content=text)
            embeddings.append(res['embedding'])
        return embeddings
        
    # Additional strict methods just in case Chroma attempts alternative calls
    def embed_query(self, input: str) -> list[float]:
        return self.__call__([input])[0]
        
    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

def query_rag_pipeline(user_query: str, db_path: str = "./db", k: int = 3) -> dict:
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = SafeGeminiEmbeddingFunction()
    
    collection = client.get_collection(
        name="document_knowledge_base",
        embedding_function=embedding_fn
    )
    
    results = collection.query(query_texts=[user_query], n_results=k)
    
    context_blocks = []
    citations = []
    
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source_name = meta['source']
        page_num = meta['page']
        citation_str = f"Source: {source_name}, Page: {page_num}"
        context_blocks.append(f"[{citation_str}]\nContext: {doc}")
        citations.append(citation_str)
        
    context_payload = "\n\n--\n\n".join(context_blocks)
    
    system_prompt = (
        "You are a professional, accurate document Q&A assistant. "
        "Answer the user's question using ONLY the provided document context below. "
        "Cite the sources (filenames and pages) inline next to facts you cite. "
        "If the answer cannot be found in the context, clearly state: "
        "'I am sorry, but the provided documents do not contain the answer to your question.' "
        "Do not make up facts or use external knowledge sources."
    )
    
    prompt = f"{system_prompt}\n\nCONTEXT INFORMATION:\n{context_payload}\n\nUSER QUESTION: {user_query}\n\nGROUNDED ANSWER:"
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content(prompt)
    
    return {
        "answer": response.text,
        "citations": citations
    }

if __name__ == "__main__":
    print("\n" + "="*50)
    print("Welcome to the Document Q&A Bot!")
    print("Type 'exit' or 'quit' to close the program.")
    print("="*50 + "\n")
    
    while True:
        question = input("\nAsk a question about your documents: ")
        if question.lower() in ['exit', 'quit']:
            break
            
        print("Searching documents and generating answer...\n")
        try:
            result = query_rag_pipeline(question)
            print("-" * 40)
            print("ANSWER:")
            print(result["answer"])
            print("-" * 40)
            print("CITATIONS USED:")
            for cite in result["citations"]:
                print(f"- {cite}")
        except Exception as e:
            print(f"An error occurred: {e}")