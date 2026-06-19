import streamlit as st
import sys
import os

# Add the src folder to the path so we can import query.py
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Try to import the query function (assuming you have a function named query_rag or similar in query.py)
# If your function is named something else, change 'query_rag' below!
try:
    from query import query_rag
except ImportError:
    st.error("Could not import the query function from query.py. Make sure you have a function that takes a question and returns an answer!")

# --- UI Layout ---
st.set_page_config(page_title="Document Q&A Bot", page_icon="📄")

st.title("📄 RAG Document Q&A Bot")
st.markdown("Ask any question, and the bot will search your vector database to find the exact answer and cite the source document!")
st.divider()

# --- Chat Interface ---
question = st.text_input("Enter your question about the documents:", placeholder="e.g., What is the 3-month roadmap for Python?")

if st.button("Generate Answer", type="primary"):
    if question:
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Call your backend logic
                answer = query_rag(question)
                
                # Display the result
                st.success("Done!")
                st.markdown("### Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question first.")