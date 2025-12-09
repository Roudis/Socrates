# Chat Interface

import chainlit as cl
import os
from app.infrastructure.llm.langchain_adapter import LangChainAdapter
from app.infrastructure.db.chroma_adapter import ChromaAdapter
from app.application.services.chat_service import SocraticChatService
from app.application.services.ingestion_service import IngestionService

# Initialize Adapters (In prod, use Dependency Injection container)
vector_store = ChromaAdapter()
llm_adapter = LangChainAdapter()
chat_service = SocraticChatService(vector_store, llm_adapter)
ingestion_service = IngestionService(vector_store)

@cl.on_chat_start
async def start():
    """Entry point for the chat session."""
    
    # Send a welcome message
    await cl.Message(content="Γεια σου! Είμαι ο Scorates. Ανέβασε το βιβλίο σου για να ξεκινήσουμε.").send()
    
    files = None
    
    # Wait for PDF upload
    while files is None:
        files = await cl.AskFileMessage(
            content="Παρακαλώ ανεβάστε το αρχείο PDF.",
            accept=["application/pdf"],
            max_size_mb=25,
            timeout=180,
        ).send()

    text_file = files
    
    # Notify user of processing
    msg = cl.Message(content=f"Επεξεργασία του {text_file.name}...")
    await msg.send()
    
    # Process PDF (Ingestion)
    # Save temp file
    temp_path = f"/tmp/{text_file.name}"
    with open(temp_path, "wb") as f:
        f.write(text_file.content)
        
    # Ingest
    material_id = await ingestion_service.ingest(temp_path)
    
    # Store material_id in user session
    cl.user_session.set("material_id", material_id)
    cl.user_session.set("history",)

    # Update message
    msg.content = f"Το αρχείο {text_file.name} είναι έτοιμο! Τι θέλεις να μελετήσουμε σήμερα;"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    """Main chat loop."""
    material_id = cl.user_session.get("material_id")
    history = cl.user_session.get("history")
    
    # Call Service
    response = await chat_service.handle_message(message.content, history, material_id)
    
    # Update History
    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": response})
    cl.user_session.set("history", history)
    
    # Stream Response
    await cl.Message(content=response).send()