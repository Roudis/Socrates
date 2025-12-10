# PDF processing logic
import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Import the port interface. In a production environment with Dependency Injection,
# this would be an abstract base class, not the concrete implementation.
from app.infrastructure.db.chroma_adapter import ChromaAdapter

class IngestionService:
    """
    Application Service responsible for the document ingestion workflow.
    
    Responsibilities:
    1. Validate and load PDF files from the filesystem.
    2. Split the raw text into overlapping chunks suitable for embedding.
    3. Delegate the storage of these chunks to the VectorStorePort.
    """

    def __init__(self, vector_store_adapter: ChromaAdapter):
        """
        Initialize the ingestion service with a database adapter.
        
        Args:
            vector_store_adapter: The interface to the vector database.
        """
        self.vector_store = vector_store_adapter
        
        # Configure the text splitter.
        # We use a hierarchical list of separators to respect natural language boundaries.
        # The overlap of 200 characters helps preserve context across splits, 
        # mitigating the risk of breaking complex Greek sentences mid-thought.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    async def process_file(self, file_path: str, metadata: Dict[str, Any] = None) -> int:
        """
        Orchestrates the processing of a single PDF file asynchronously.
        
        This method offloads the blocking file I/O and CPU-bound text splitting
        to a separate thread to avoid blocking the main asyncio event loop.
        """
        import asyncio # Import locally to avoid top-level changes
        loop = asyncio.get_running_loop()
        # Execute the synchronous processing in a thread pool
        return await loop.run_in_executor(
            None, 
            self._process_file_sync, 
            file_path, 
            metadata
        )

    def _process_file_sync(self, file_path: str, metadata: Dict[str, Any] = None) -> int:
        """
        Synchronous implementation of file processing.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")

        # 1. Load: Extract text from the PDF
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()

        # Update the metadata for each page/document extracted
        if metadata:
            for doc in raw_documents:
                doc.metadata.update(metadata)

        # 2. Split: Chunk the text
        chunks = self.text_splitter.split_documents(raw_documents)

        # 3. Store: Persist to the vector database in batches
        # Chroma handles the embedding generation internally via the function passed to it.
        # We batch strictly to 1 to minimize load on the local Ollama instance.
        import time
        BATCH_SIZE = 1 
        if chunks:
            total_chunks = len(chunks)
            for i in range(0, total_chunks, BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                
                # Retry mechanism for unstable embedding service
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.vector_store.add_documents(batch)
                        time.sleep(0.5) # Small buffer to let the server breathe
                        break # Success, move to next batch
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"Failed to ingest batch {i} after {max_retries} attempts: {e}")
                            raise e
                        print(f"Batch {i} failed (attempt {attempt+1}/{max_retries}), retrying in 2s... Error: {e}")
                        time.sleep(2)

        return len(chunks)

    def clear_database(self) -> None:
        """
        Resets the knowledge base.
        
        This forwards the reset command to the adapter, effectively wiping
        the curriculum data to allow for a fresh start.
        """
        self.vector_store.reset()