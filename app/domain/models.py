# Pydantic entities (StudyMaterial, Message)

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from uuid import UUID, uuid4
from datetime import datetime

class StudyMaterial(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    content_hash: str  # To prevent duplicate uploads
    upload_date: datetime = Field(default_factory=datetime.now)
    processed: bool = False

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: Optional[List[str]] = None  # References to PDF pages

class ConversationHistory(BaseModel):
    session_id: str
    messages: List[ChatMessage] =

    def add_message(self, role: str, content: str, sources: List[str] = None):
        self.messages.append(ChatMessage(role=role, content=content, sources=sources))