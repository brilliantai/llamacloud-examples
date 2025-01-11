import os
import gradio as gr
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

###############################################################################
#                           SETUP & GLOBAL CONFIG
###############################################################################

# --- LlamaCloud / OpenAI-Compat Client ---
LLAMACLOUD_API_KEY = os.getenv("LLAMACLOUD_API_KEY")  # Replace with your real key
client = OpenAI(
    base_url="https://api.llamacloud.co",
    api_key=LLAMACLOUD_API_KEY
)

# --- Embedding Model ---
embedding_model = "snowflake-arctic-embed-l-v2.0"

# --- Qdrant Setup ---
#  You can point to a local disk-based DB via `path="vector_store"` 
#  or to a running Qdrant server via `url="http://localhost:6333"`.
qdrant_client = QdrantClient(path="vector_store")  # local disk-based
collection_name = "rag_collection"

# Ensure the collection exists (create if needed)
def initialize_qdrant_collection():
    try:
        # Check if collection exists
        qdrant_client.get_collection(collection_name)
    except:
        # If not found, create one with an appropriate vector size/distance
        qdrant_client.create_collection(
            collection_name,
            vectors_config=VectorParams(
                size=1024,          # Must match your embedding dimension
                distance=Distance.COSINE
            )
        )

initialize_qdrant_collection()


###############################################################################
#                           HELPER FUNCTIONS
###############################################################################

def read_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    with open(file_path, 'rb') as file:
        pdf_document = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        return text

def read_internet_data(url: str) -> str:
    """Extract text content from a webpage."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove non-content elements
        for tag in soup(["script", "style", "head", "title", "meta"]):
            tag.extract()
        
        # Clean and format text
        text = soup.get_text(separator="\n").strip()
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())

    except Exception as e:
        print(f"Error processing {url}: {e}")
        return ""

def is_url(path_or_url: str) -> bool:
    """Check if string is a valid URL."""
    try:
        result = urlparse(path_or_url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except ValueError:
        return False

def chunk_data(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """Split text into overlapping chunks using a LangChain text splitter."""
    documents = [Document(page_content=text)]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return [doc.page_content for doc in chunks]

def process_and_embed(source: str):
    """Process a source (PDF file path or URL) and store embeddings in Qdrant."""
    # 1) Read content
    if is_url(source):
        text = read_internet_data(source)
    else:
        text = read_pdf(source)
    
    # 2) Chunk data
    chunks = chunk_data(text)
    
    # 3) Generate embeddings for each chunk
    embeddings = client.embeddings.create(
        model=embedding_model,
        input=chunks
    )
    
    # 4) Upsert each chunk + embedding into Qdrant
    points = [
        PointStruct(
            id=idx,                     # numeric ID or a unique ID
            vector=data.embedding,      # the embedding vector
            payload={"text": txt}       # store the chunk text
        )
        for idx, (data, txt) in enumerate(zip(embeddings.data, chunks))
    ]
    
    qdrant_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )

def retrieve_context(query: str, top_k: int = 5) -> list:
    """Retrieve the most relevant chunks from Qdrant for the given query."""
    # Embed the query
    query_embedding = client.embeddings.create(
        input=query,
        model=embedding_model
    ).data[0].embedding
    
    # Vector search in Qdrant
    results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=top_k
    )
    
    return [result.payload["text"] for result in results]

def generate_response(query: str, context_chunks: list) -> str:
    """Use the retrieved context + LLM to generate a response."""
    # Define a prompt template
    template = """
    Based on the following context:

    {context}

    Answer this question:
    {query}

    Provide a clear and direct answer without referencing the context.
    """
    prompt = PromptTemplate.from_template(template)
    
    # Create augmented query
    augmented_query = prompt.format(
        context="\n\n".join(context_chunks),
        query=query
    )
    
    # Initialize the LLM
    llm = ChatOpenAI(
        base_url="https://api.llamacloud.co",
        api_key=LLAMACLOUD_API_KEY,
        model="llama-3.3-70b",
        temperature=0.7,
        max_tokens=500
    )
    
    # Generate and parse the answer
    response = llm.invoke(augmented_query)
    final_answer = StrOutputParser().invoke(response)
    return final_answer


###############################################################################
#                           RAG PIPELINE FOR GRADIO
###############################################################################

def rag_pipeline(pdf_file, url, user_query):
    """
    1) If a PDF file is provided, ingest it into Qdrant
    2) Else if a URL is provided, ingest it
    3) Then retrieve context from Qdrant for the user query
    4) Generate final answer
    5) Return answer + context
    """
    # Step 1: Check for inputs
    did_embed = False
    if pdf_file is not None:
        # Save temp PDF
        temp_pdf_path = pdf_file.name
        process_and_embed(temp_pdf_path)
        did_embed = True
    elif url.strip():
        process_and_embed(url.strip())
        did_embed = True
    
    # If user didn't upload or enter anything, just proceed with the existing data in Qdrant
    if not user_query.strip():
        return "No query provided.", "No context."

    # Step 2: Retrieve relevant chunks
    context_chunks = retrieve_context(user_query, top_k=3)

    # Step 3: Generate final answer
    answer = generate_response(user_query, context_chunks)

    # Return the LLM's answer + the context used
    # Join context chunks for display
    context_display = "\n---\n".join(context_chunks)
    return answer, context_display


###############################################################################
#                           BUILD GRADIO APP
###############################################################################

def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("# LlamaCloud RAG Demo")

        with gr.Row():
            # Left Pane
            with gr.Column():
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                url_input = gr.Textbox(label="Enter URL here")
                user_query = gr.Textbox(label="Ask a question", lines=2)

                # Button to trigger ingestion + answer
                submit_btn = gr.Button("Get Answer")

            # Right Pane
            with gr.Column():
                answer_box = gr.Textbox(label="Answer", lines=4)
                context_box = gr.Textbox(label="Context used", lines=8)

        # When the button is clicked, run rag_pipeline
        submit_btn.click(
            fn=rag_pipeline,
            inputs=[pdf_input, url_input, user_query],
            outputs=[answer_box, context_box]
        )

    return demo


###############################################################################
#                               MAIN
###############################################################################

if __name__ == "__main__":
    # Build and launch the Gradio app
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
