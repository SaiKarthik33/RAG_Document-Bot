# 📄 RAG Document Q&A Bot

**1. Project Overview**
This is a complete Retrieval-Augmented Generation (RAG) pipeline built to query local documents. It ingests PDF files, chunks the text, stores the embeddings in a local vector database, and uses a generative LLM to provide grounded answers with exact source citations.

**2. Tech Stack**
* **Language:** Python 3.12
* **LLM & Embeddings:** Google Gemini API (`gemini-2.5-flash` for generation, `gemini-embedding-001` for embeddings)
* **Vector Database:** ChromaDB (Persistent local storage)
* **Document Processing:** `pypdf`
* **Frontend:** Streamlit Community Cloud

**3. Architecture Overview**
1. **Ingestion:** Reads raw PDF files from the `/data` directory.
2. **Chunking:** Splits text into manageable, overlapping segments.
3. **Storage:** Generates embeddings via Gemini and persists them locally using ChromaDB.
4. **Retrieval:** Embeds the user's query and performs a similarity search to fetch the top 3 most relevant context chunks.
5. **Generation:** Passes the retrieved chunks to the Gemini LLM with a strict system prompt to answer *only* using the provided context and output citations.

**4. Chunking Strategy**
I implemented a **Fixed-Size Chunking** strategy (1,000 characters with a 200-character overlap). 
* *Justification:* This ensures that chunks are small enough to stay within the LLM's context window while the 200-character overlap prevents cutting off important concepts or sentences mid-thought.

**5. Embedding Model & Vector Database Choice**
* *Vector DB:* I chose **ChromaDB**. It is open-source, runs locally without needing external cloud configuration, and allows for persistent disk storage, making it perfect for rapid prototyping and Streamlit integration.
* *Embedding:* I utilized a custom wrapper (`SafeGeminiEmbeddingFunction`) to connect the **Google Gemini Embedding Model** to ChromaDB to ensure high-dimensional semantic accuracy.

**6. Setup Instructions**
1. Clone the repository:
   `git clone https://github.com/saikarthik33/rag_document-bot.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Run the ingestion script to build the database:
   `python src/ingest.py`
4. Launch the Streamlit application:
   `streamlit run src/main.py`

**7. Environment Variables**
Create a `.env` file in the root directory and add your Google Gemini API key:
`GEMINI_API_KEY="your_api_key_here"`

**8. Example Queries**
1. *[Replace with a question about your doc 1]*
2. *[Replace with a question about your doc 2]*
3. *[Replace with a cross-document question]*
4. *[Replace with a specific detail question]*
5. *[Replace with another question]*

**9. Known Limitations**
* The bot currently only supports PDF formats; DOCX and TXT ingestion would require additional parsing logic.
* Because ChromaDB runs locally in this implementation, deploying to a stateless cloud environment (like Streamlit Cloud) required pushing the pre-built `db` folder to GitHub.
