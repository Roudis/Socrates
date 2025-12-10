# PDF extraction

import fitz  # PyMuPDF
import unicodedata
from app.domain.ports import DocumentLoaderPort
from langchain_core.documents import Document
from typing import List

class PyMuPDFLoaderAdapter(DocumentLoaderPort):
    def load_pdf(self, file_path: str) -> List:
        doc = fitz.open(file_path)
        documents = []
        
        for i, page in enumerate(doc):
            # Extract text with flags to handle whitespace correctly
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            
            # CRITICAL: Normalize Greek characters to NFC
            # This fixes issues where 'ά' is stored as 'α' + '´'
            text = unicodedata.normalize("NFC", text)
            
            if text.strip():
                # Metadata is crucial for citations
                metadata = {"source": file_path, "page": i + 1}
                documents.append(Document(page_content=text, metadata=metadata))
                
        return documents