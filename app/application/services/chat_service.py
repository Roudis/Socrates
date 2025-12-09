# Socratic logic orchestration

from typing import List
from app.domain.ports import VectorStorePort, LLMPort
from app.domain.models import ChatMessage
from app.infrastructure.llm.prompts import SOCRATIC_SYSTEM_PROMPT_EL

class SocraticChatService:
    def __init__(self, vector_store: VectorStorePort, llm: LLMPort):
        self.vector_store = vector_store
        self.llm = llm

    async def handle_message(self, message: str, history: List[ChatMessage], material_id: str) -> str:
        # 1. Retrieve Context
        # Query the vector store for the top 3 relevant chunks
        docs = self.vector_store.similarity_search(message, material_id, k=3)
        
        if not docs:
            return "Δεν βρήκα σχετικές πληροφορίες στο βιβλίο για αυτό το θέμα. Μπορείς να γίνεις πιο συγκεκριμένος;"

        context_text = "\n\n".join([f"[Σελίδα {d.metadata['page']}]: {d.page_content}" for d in docs])

        # 2. Format History
        # Flatten history for the prompt
        formatted_history = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in history[-5:]]) # Keep last 5 turns
        
        # 3. Construct System Prompt
        full_prompt = SOCRATIC_SYSTEM_PROMPT_EL.format(
            context=context_text,
            chat_history=formatted_history,
            question=message
        )

        # 4. Generate Socratic Response
        response = await self.llm.generate_response(full_prompt)
        
        return response