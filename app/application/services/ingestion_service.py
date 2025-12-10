# PDF processing logic
import os
import time
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.infrastructure.db.chroma_adapter import ChromaAdapter

class IngestionService:
    def __init__(self, vector_store_adapter: ChromaAdapter):
        self.vector_store = vector_store_adapter
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    async def process_file(self, file_path: str, metadata: Dict[str, Any] = None) -> int:
        import asyncio 
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, 
            self._process_file_sync, 
            file_path, 
            metadata
        )

    def _process_file_sync(self, file_path: str, metadata: Dict[str, Any] = None) -> int:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at path: {file_path}")

        # 1. Load
        loader = PyPDFLoader(file_path)
        raw_documents = loader.load()

        if metadata:
            for doc in raw_documents:
                doc.metadata.update(metadata)

        # 2. Split
        chunks = self.text_splitter.split_documents(raw_documents)

        # 3. Store
        # FAULT TOLERANT MODE
        # Batch size 10 is safe for M1 Max.
        BATCH_SIZE = 10
        successful_chunks = 0
        
        if chunks:
            total_chunks = len(chunks)
            for i in range(0, total_chunks, BATCH_SIZE):
                batch = chunks[i : i + BATCH_SIZE]
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        self.vector_store.add_documents(batch)
                        successful_chunks += len(batch)
                        
                        # Small pause to keep memory usage stable
                        time.sleep(0.5) 
                        break 
                    except Exception as e:
                        # IF ALL RETRIES FAIL:
                        if attempt == max_retries - 1:
                            # Log the error but DO NOT crash the app.
                            print(f"⚠️ SKIPPING BATCH {i} (Chunks {i}-{i+BATCH_SIZE}). Error: {e}")
                            # We 'break' out of the retry loop and continue to the next batch
                            break
                        
                        print(f"Batch {i} failed (attempt {attempt+1}), retrying... Error: {e}")
                        time.sleep(2.0)

        return successful_chunks

    def clear_database(self) -> None:
        self.vector_store.reset()