import os
import uuid
import json
import requests
import fitz
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI

###############################################################################
# Helper functions: read_pdf, is_url, read_internet_data
###############################################################################
def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    with open(file_path, 'rb') as file:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text

def is_url(path_or_url: str) -> bool:
    """Check if string is a valid URL (http/https)."""
    try:
        result = urlparse(path_or_url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False

def read_internet_data(url: str) -> str:
    """Read text content from a URL."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    else:
        raise ValueError(f"Failed to retrieve data from {url}")

###############################################################################
# Simple chunking utility
###############################################################################
def chunk_text_into_passages(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        if start < 0:
            break
    return chunks

###############################################################################
# Qdrant Setup
###############################################################################
if "qdrant_client" not in st.session_state:
    st.session_state["qdrant_client"] = QdrantClient(path="vector_store")

qdrant_client = st.session_state["qdrant_client"]
collection_name = "rag_collection"

collections_info = qdrant_client.get_collections().collections
existing_collections = collections_info or []
collection_names = [c.name for c in existing_collections if c is not None]

if collection_name not in collection_names:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=1024,
            distance=Distance.COSINE
        )
    )

###############################################################################
# Document processing & embedding
###############################################################################
def process_and_embed(
    source: str,
    embedding_client: OpenAI,
    embedding_model: str = "snowflake-arctic-embed-l-v2.0"
) -> str:
    """Reads PDF or URL, chunks text, embeds chunks, and upserts to Qdrant."""
    # Decide if it's PDF or URL
    if source.lower().endswith(".pdf"):
        base_name = os.path.basename(source)
        doc_id = os.path.splitext(base_name)[0]
        text = read_pdf(source)
    elif is_url(source):
        doc_id = source
        text = read_internet_data(source)
    else:
        raise ValueError("Source must be a PDF file path or URL")

    # Chunk the text
    chunks = chunk_text_into_passages(text)

    # Create embeddings
    embeddings = embedding_client.embeddings.create(
        model=embedding_model,
        input=chunks
    )

    # Prepare Qdrant points
    points = []
    for chunk_text, embed_data in zip(chunks, embeddings.data):
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=embed_data.embedding,
                payload={"text": chunk_text, "doc_id": doc_id}
            )
        )

    # Upsert into Qdrant
    qdrant_client.upsert(collection_name=collection_name, points=points)
    return doc_id

def retrieve_context(query: str, embedding_client: OpenAI, top_k: int = 3) -> str:
    """Generate an embedding for the query, then search in Qdrant for top_k matches."""
    embedding_model = "snowflake-arctic-embed-l-v2.0"
    query_embed = embedding_client.embeddings.create(
        model=embedding_model,
        input=query
    ).data[0].embedding

    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embed,
        limit=top_k
    )

    context_snippets = [res.payload["text"] for res in results]
    context = "\n\n".join(context_snippets)
    return context

###############################################################################
# Chatbot class with conversation history for multi-turn
###############################################################################
class Chatbot:
    def __init__(self, model_name: str):
        """
        model_name: e.g. "mistral-nemo", "llama-3.2-8b", etc.
        """
        # Load API key from environment
        api_key = os.getenv("LLAMACLOUD_API_KEY")
        if not api_key:
            raise ValueError("Please set 'LLAMACLOUD_API_KEY' in environment variables.")

        self.model_name = model_name
        self.client = OpenAI(base_url="https://api.llamacloud.co", api_key=api_key)
        # Keep track of conversation messages. This will be passed each time we call .create().
        self.conversation_history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def chat(self, user_message: str) -> str:
        """
        Multi-turn chat:
          1) Retrieve relevant context from Qdrant.
          2) Add system message with that context.
          3) Add the user's new message.
          4) Send the entire conversation to the LLM.
        """
        # 1) Retrieve context for the current user query
        context = retrieve_context(query=user_message, embedding_client=self.client, top_k=3)

        # 2) Add a system message that includes the retrieved context
        context_message = (
            "Relevant context to help answer the user's question:\n\n"
            f"{context}\n\n"
            "Provide a concise and direct answer without revealing the context to the user."
        )
        self.add_message("system", context_message)

        # 3) Add the user's new message
        self.add_message("user", user_message)

        # 4) Send the entire conversation history to the model
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                temperature=0.7
            )
            assistant_message = response.choices[0].message.content

            # Add the model's answer to conversation history
            self.add_message("assistant", assistant_message)
            return assistant_message
        except Exception as e:
            print(f"Error during chat: {str(e)}")
            return "I'm sorry, I encountered an error."

###############################################################################
# STREAMLIT UI
###############################################################################
def main():
    st.title("RAG Chat with Documents - Multi-turn Demo")

    ############################################################################
    # 1. Model selection & chatbot creation
    ############################################################################
    model_list = ["mistral-nemo", "llama-3.2-8b"]
    chosen_model = st.selectbox("Select a Model:", model_list)

    # Create a Chatbot instance if none exists
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = Chatbot(model_name=chosen_model)

    # If the user changes the model, re-init the Chatbot
    if chosen_model != st.session_state["chatbot"].model_name:
        st.session_state["chatbot"] = Chatbot(model_name=chosen_model)

    # This is the embedding client for any doc ingestion
    embedding_client = st.session_state["chatbot"].client

    ############################################################################
    # 2. Document upload & ingestion
    ############################################################################
    uploaded_file = st.file_uploader("Upload a PDF to embed into the vector store", type=["pdf"])
    if uploaded_file is not None:
        temp_file_path = f"/tmp/{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"Saved file to {temp_file_path}")

        if st.button("Ingest Document"):
            doc_id = process_and_embed(
                source=temp_file_path,
                embedding_client=embedding_client,
                embedding_model="snowflake-arctic-embed-l-v2.0"
            )
            st.success(f"Document '{doc_id}' has been ingested into Qdrant.")
    else:
        st.warning("Please upload a PDF file.")

    # Show the Qdrant collection stats
    if st.button("Show Vector Store Stats"):
        collection_info = qdrant_client.get_collection(collection_name)
        st.write(f"Collection '{collection_name}' has {collection_info.vectors_count} vectors.")

    ############################################################################
    # 3. Multi-turn Chat with the Document
    ############################################################################
    st.write("## Multi-turn Chat")

    # We also keep a separate local 'chat_history' just for displaying in Streamlit.
    # The Chatbot class has its own conversation_history that gets passed to the LLM.
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("Ask a question about the ingested document:")

    if st.button("Send Message") and user_input.strip():
        # Chat with the bot (multi-turn)
        bot_response = st.session_state["chatbot"].chat(user_input)

        # Display messages in the Streamlit UI
        st.session_state["chat_history"].append(("user", user_input))
        st.session_state["chat_history"].append(("assistant", bot_response))

    # Show the conversation
    for role, content in st.session_state["chat_history"]:
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Assistant:** {content}")

if __name__ == "__main__":
    main()
