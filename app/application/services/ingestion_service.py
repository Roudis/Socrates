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
        Orchestrates the processing of a single PDF file.
        
        Args:
            file_path: The absolute path to the temporary file on disk.
            metadata: A dictionary of additional metadata (e.g., filename, uploader)
                      to be attached to every chunk generated from this file.
            
        Returns:
            int: The total number of chunks created and stored.
            
        Raises:
            FileNotFoundError: If the provided path does not exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")

        # 1. Load: Extract text from the PDF
        # PyPDFLoader is a reliable choice for standard text-based PDFs.
        # For OCR-heavy PDFs, an integration with 'unstructured' or Tesseract would be needed.
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()

        # Update the metadata for each page/document extracted
        if metadata:
            for doc in raw_documents:
                doc.metadata.update(metadata)

        # 2. Split: Chunk the text
        chunks = self.text_splitter.split_documents(raw_documents)

        # 3. Store: Persist to the vector database
        # Chroma handles the embedding generation internally via the function passed to it.
        if chunks:
            self.vector_store.add_documents(chunks)

        return len(chunks)

    def clear_database(self) -> None:
        """
        Resets the knowledge base.
        
        This forwards the reset command to the adapter, effectively wiping
        the curriculum data to allow for a fresh start.
        """
        self.vector_store.reset()