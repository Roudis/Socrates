# Interfaces (ABCs)

from abc import ABC, abstractmethod
from typing import List, Any
from.models import StudyMaterial, ChatMessage

class DocumentLoaderPort(ABC):
    @abstractmethod
    def load_pdf(self, file_path: str) -> List[Any]:
        """Extracts text and metadata from PDF."""
        pass

class VectorStorePort(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Any], material_id: str):
        """Embeds and stores document chunks."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, material_id: str, k: int = 4) -> List[Any]:
        """Retrieves relevant context for a specific material."""
        pass

class LLMPort(ABC):
    @abstractmethod
    async def generate_response(self, messages: List[ChatMessage], context: str) -> str:
        """Generates a response based on history and context."""
        pass