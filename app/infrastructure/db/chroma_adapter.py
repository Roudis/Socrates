# Vector Store implementation

import os
import shutil
from typing import List, Optional, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings

# In a strict Clean Architecture, the 'Document' class would be a Domain entity.
# The adapter would be responsible for mapping the Domain Document to the 
# LangChain/Chroma specific document format. For the purpose of this report 
# and typical Pythonic pragmatism, we assume the Application layer utilizes 
# the LangChain Document schema or a compatible Protocol.

class ChromaAdapter:
    """
    Adapter for ChromaDB implementing the VectorStorePort.
    
    This class encapsulates all interactions with the Chroma vector database,
    providing a clean API for adding documents, performing similarity searches,
    and managing the persistence of the vector index. It isolates the 
    application from the specifics of the 'langchain_chroma' library.
    """

    def __init__(
        self, 
        collection_name: str, 
        embedding_function: Embeddings,
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the Chroma adapter with persistence configuration.
        
        Args:
            collection_name: The namespace for the dataset (e.g., 'greek_curriculum').
                             Separating collections allows for multi-tenancy or 
                             subject-specific isolation in the future.
            embedding_function: The LangChain-compatible embedding model instance.
                                This performs the text-to-vector transformation.
            persist_directory: The local filesystem path where the database 
                               files will be stored. This should be mapped to a 
                               Docker volume for data durability.
        """
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        
        # Initialize the Chroma client.
        # The client automatically handles loading existing data from the 
        # persist_directory if it exists, or creates a new database if it doesn't.
        self._vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List) -> None:
        """
        Ingests a list of documents into the vector store.
        
        This method handles the vectorization (via the embedding_function)
        and storage of the document content and metadata.
        
        Args:
            documents: A list of Document objects containing text and metadata.
        """
        if not documents:
            return
        
        # Chroma handles batching internally, but for extremely large datasets,
        # manual batching might be implemented here in the future.
        self._vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> List:
        """
        Performs a semantic similarity search against the vector index.
        
        Args:
            query: The user's natural language query.
            k: The number of relevant documents to retrieve.
            
        Returns:
            A list of the k most similar documents.
        """
        return self._vector_store.similarity_search(query, k=k)

    def as_retriever(self, search_type: str = "similarity", search_kwargs: dict = None) -> Any:
        """
        Exposes the vector store as a LangChain Retriever interface.
        
        This is crucial for integration with LangChain's retrieval chains,
        which expect a Retriever object rather than a raw vector store.
        
        Args:
            search_type: The type of search (e.g., "similarity", "mmr").
            search_kwargs: Additional arguments like 'k' or 'score_threshold'.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 4}
            
        return self._vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def reset(self) -> None:
        """
        Completely clears the database and resets the state.
        
        This method is useful for development iterations or re-ingestion 
        workflows where the curriculum has changed significantly.
        """
        # Chroma's API has varied in how it handles deletion. 
        # A filesystem-level removal offers the most robust 'hard reset'.
        if os.path.exists(self.persist_directory):
            # Attempt to release the internal client reference to avoid file locks
            self._vector_store = None 
            try:
                shutil.rmtree(self.persist_directory)
            except OSError as e:
                print(f"Error removing persistence directory: {e}")
            
            # Re-initialize the client to create a fresh, empty database
            try:
                self._vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory
                )
            except Exception as e:
                # If re-init fails (e.g. race condition), log and retry or pass
                print(f"Warning: Failed to re-initialize Chroma after reset: {e}")
                pass