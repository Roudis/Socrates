import os
import chainlit as cl
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Internal Imports - connecting the layers
from app.infrastructure.db.chroma_adapter import ChromaAdapter
from app.infrastructure.llm.langchain_adapter import LangChainAdapter
from app.application.services.ingestion_service import IngestionService

# --- Configuration & Initialization ---
CHROMA_PATH = "/home/scorates/chroma_db"
MODEL_NAME = os.getenv("MODEL_NAME", "ilsp/llama-krikri-8b-instruct")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Initialize Embedding Model
# We use a robust multilingual model to handle Greek text effectively.
# This runs locally within the app container.
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# Initialize Infrastructure Adapters
chroma_adapter = ChromaAdapter(
    collection_name="greek_curriculum",
    embedding_function=embedding_model,
    persist_directory=CHROMA_PATH
)

llm_adapter = LangChainAdapter(
    model_name=MODEL_NAME,
    base_url=OLLAMA_URL,
    temperature=0.1, # Low temperature for groundedness
    context_window=8192 # Matched to Docker config
)

# Initialize Application Service
ingestion_service = IngestionService(chroma_adapter)

@cl.on_chat_start
async def start():
    """
    Session Initialization Hook.
    Sets up the user session and sends the welcome message.
    """
    welcome_msg = """
    **Γεια σας! I am Scorates.** 
    
    I am your Socratic Tutor for the Greek curriculum. 
    Please upload a PDF textbook or notes to begin, or ask me a question about the material.
    """
    await cl.Message(content=welcome_msg).send()
    
    # Store references in the session for potential stateful operations later
    cl.user_session.set("vector_store", chroma_adapter)
    cl.user_session.set("llm", llm_adapter)

@cl.on_message
async def main(message: cl.Message):
    """
    Main Event Loop.
    Handles both document ingestion (if files are present) and RAG chat.
    """
    
    # 1. Handle File Uploads
    # Chainlit attaches files to the message object.
    if message.elements:
        # Filter for PDF files
        files = [file for file in message.elements if "pdf" in file.mime]
        if files:
            msg = cl.Message(content=f"Processing {len(files)} file(s)...")
            await msg.send()
            
            total_chunks = 0
            for file in files:
                # Chainlit saves temp files to `file.path`. 
                # We pass this path to our Domain Service.
                chunks = await ingestion_service.process_file(
                    file.path, 
                    metadata={"source": file.name, "user_id": cl.user_session.get("id")}
                )
                total_chunks += chunks
            
            await cl.Message(content=f"Ingestion complete. Added {total_chunks} chunks to the knowledge base.").send()
            
            # If the user only uploaded files and sent no text, return early.
            if not message.content:
                return

    # 2. Handle RAG Chat
    # Define the Socratic Prompt Template adhering to Llama 3 format.
    # We use Greek instructions to align with the model's training.
    template = """<|start_header_id|>system<|end_header_id|>
You are Scorates, a Socratic Tutor for the Greek curriculum. 
Use the following pieces of context to answer the user's question. 
Do not give the answer directly. Instead, ask guiding questions to help the student find the answer.
If the answer is not in the context, say you don't know, but try to guide them based on general knowledge.
Respond in Greek unless asked otherwise.

Context: {context}<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Convert the adapter to a Retriever interface for the chain
    retriever = chroma_adapter.as_retriever(search_kwargs={"k": 3})
    
    # Instantiate the RetrievalQA Chain
    # We use 'stuff' chain type which fits all context into one prompt.
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_adapter.get_llm_instance(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    # Execute the Chain asynchronously
    # The callback handler enables Chainlit to show the "Thought Process" UI
    res = await qa_chain.acall(
        message.content, 
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    
    answer = res["result"]
    source_documents = res["source_documents"]

    # Format the Source Documents for the UI
    text_elements = []
    if source_documents:
        for idx, source in enumerate(source_documents):
            source_name = source.metadata.get("source", "Unknown")
            # Create a text element for each source chunk
            text_elements.append(
                cl.Text(
                    content=source.page_content, 
                    name=f"Source {idx+1} ({source_name})",
                    display="inline"
                )
            )

    # Send the Final Response
    await cl.Message(content=answer, elements=text_elements).send()