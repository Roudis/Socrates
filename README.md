Technical Specification and Architectural Blueprint for "Scorates": A Socratic AI Tutor for the Greek Secondary Education System
Executive Summary
The intersection of Artificial Intelligence and education currently stands at a critical juncture. While Large Language Models (LLMs) offer unprecedented access to information, their standard deployment as direct-answer engines poses a significant risk to cognitive development, particularly in formative educational years. The prevailing "search-and-retrieve" paradigm encourages surface-level engagement, potentially atrophying the very critical thinking skills education aims to cultivate. This report outlines the comprehensive technical and pedagogical design for Scorates, a study support application specifically engineered for the Greek secondary education system (Gymnasio and Lykeio).

Scorates is not a chatbot in the conventional sense; it is a Digital Maieutic Engine. Deriving its core philosophy from the Socratic method (maieutikos, the art of midwifery), the system is designed to "birth" knowledge from the student rather than delivering it. By integrating advanced Retrieval-Augmented Generation (RAG) architectures with a strictly constrained pedagogical persona, Scorates transforms the interaction from passive consumption to active inquiry.

This document serves as an exhaustive blueprint for software architects, educational technologists, and engineering teams. It details the selection of Greek-optimized Natural Language Processing (NLP) models, specifically addressing the linguistic complexities of Modern Greek. It specifies a robust, clean layered architecture using Python, FastAPI, and LangChain to ensure modularity and scalability. Furthermore, it provides the precise prompting engineering required to enforce Socratic dialogue in Greek and the infrastructure necessary to deploy this system securely and efficiently using Docker.

The architecture proposed herein prioritizes pedagogical integrity over speed of answer delivery, architectural cleanliness over rapid prototyping, and linguistic precision over generic multilingual support. It represents a paradigm shift from "AI as Oracle" to "AI as Tutor."

1. Pedagogical Framework: The Digital Socratic Method
1.1. The Crisis of the "Answer Engine" in Education
The integration of Generative AI into educational settings has largely followed a trajectory of efficiency. Tools like ChatGPT, Claude, or Gemini are optimized to satisfy user intent as quickly and accurately as possible. If a student asks, "What were the causes of the Greek War of Independence?", these models provide a synthesized list of historical, economic, and social factors. While efficient, this interaction bypasses the cognitive struggle—the "desirable difficulty"—required for deep learning.   

In the context of the Greek educational system, which is historically text-heavy and examination-focused, this creates a specific hazard. Students are often conditioned to memorize text. An AI that summarizes text reinforces rote memorization. To counter this, an EdTech intervention must fundamentally alter the loop of interaction. It must refuse to be an encyclopedia and insist on being a dialectic partner.

1.2. Maieutics: The Theoretical Foundation
The Socratic method operates on the premise that the student already possesses the latent capacity for understanding and that the teacher's role is to guide the student to realize this understanding through questioning. This process involves several distinct cognitive movements which Scorates must emulate:

Elenchus (Refutation): The system must identify contradictions or gaps in the student's initial premise.

Aporia (Perplexity): The system must guide the student to a state of acknowledged ignorance or curiosity, creating the fertile ground for new knowledge.

Maieutics (Birthing): The system assists the student in constructing the correct answer based on their own reasoning and the available text.

Implementing this in software requires a stateful pedagogical engine. The AI cannot simply react to the last message; it must maintain a model of the student's current understanding relative to the target concept.   

1.2.1. The "Guide-on-the-Side" Paradigm
Scorates adopts the "Guide-on-the-Side" persona. This dictates specific constraints on the system prompt:

Prohibition of Direct Answers: Under no circumstances should the system provide the final answer to a curricular question if that answer can be derived from the text.

Contextual Anchoring: Every question asked by the AI must be solvable using the uploaded study material. Asking questions beyond the scope of the provided text frustrates the student and breaks the pedagogical contract.   

Incremental Complexity: The dialogue must move from low-order cognitive skills (remembering, understanding) to high-order skills (analyzing, evaluating), mirroring Bloom's Taxonomy.

1.3. Adaptation for the Greek Curriculum
The Greek curriculum (Gymnasio/Lykeio) is standardized. Students utilize specific textbooks mandated by the Ministry of Education (ITYE - Diophantus). Unlike general knowledge queries, students using Scorates will likely be studying for specific Panhellenic examinations or in-school tests based on these specific texts.

Therefore, Scorates acts as a "Closed-Book" tutor relative to the world but an "Open-Book" tutor relative to the uploaded PDF. It must resist utilizing its pre-trained knowledge base (e.g., knowledge about the American Civil War when the topic is the French Revolution) and adhere strictly to the uploaded Greek text to ensure alignment with exam requirements. The interface and reasoning must use Modern Greek (Demotic), handling the specific academic vocabulary found in Greek school textbooks.   

2. Natural Language Processing Strategy for Modern Greek
Building a high-performance RAG system for Greek requires navigating a landscape where resources are significantly scarcer than for English. The choice of embedding models, tokenizers, and chunking strategies is foundational to the system's success.

2.1. The Linguistic Challenges of Greek NLP
Modern Greek presents specific hurdles for standard NLP pipelines designed primarily for English:

Morphological Richness: Greek is a highly inflected language. Nouns have three genders, four cases, and two numbers. Verbs have complex conjugation patterns. A keyword search for "student" (μαθητής) might miss "students" (μαθητές) or "of the student" (μαθητή). While stemming helps, semantic vector search is far superior—provided the embedding model understands these relationships.   

Alphabet and Encoding: The Greek alphabet (Ελληνικό αλφάβητο) occupies a specific range in Unicode (U+0370 to U+03FF). However, legacy PDFs often use disparate encodings or "precomposed" characters (e.g., ά as a single code point vs. α + ´). This creates a "mojibake" risk during text extraction where the visual representation is correct, but the underlying bytes do not match the embedding model's vocabulary.   

Tokenization Efficiency: Most LLM tokenizers (like OpenAI's cl100k_base or Llama's SentencePiece) are optimized for English. Greek text is "expensive" in terms of tokens. A typical Greek word is often split into multiple tokens, whereas an English word is usually one. This impacts the context window limits of the embedding models (often limited to 512 tokens) and the cost of LLM inference.

2.2. Evaluation of Greek-Compatible Embedding Models
The effectiveness of the RAG pipeline depends entirely on the quality of the vector representations. If the system cannot find the relevant paragraph in the textbook, the Socratic engine has no material to work with. We evaluated top open-source candidates based on the Massive Text Embedding Benchmark (MTEB) and their specific applicability to Greek.   

2.2.1. Candidate 1: Greek-Specific BERT Models (dimitriz/st-greek-media-bert-base-uncased)
This model is a fine-tuned version of BERT specifically for the Greek language, trained on a dataset of Greek media, internet, and social media text.

Architecture: BERT-base (12 layers, 768 hidden size).

Pros: High sensitivity to Greek cultural nuances and slang. Excellent for classification tasks within the Greek domain.

Cons: Strict 512-token limit. Being a BERT-based model, it relies on older pre-training objectives and lacks "instruction tuning." It treats queries and documents symmetrically, which is suboptimal for RAG where queries are short questions and documents are long text passages.   

2.2.2. Candidate 2: Multilingual E5 (intfloat/multilingual-e5-large-instruct)
The E5 (Embeddings from the Encoder) family currently represents the state-of-the-art in open-source multilingual embeddings. The "instruct" variant is particularly crucial.

Architecture: XLM-RoBERTa backbone, 24 layers, 1024 embedding dimension.

Training Data: Trained on massive multilingual datasets (mC4, Wikipedia) including significant Greek corpora.

Mechanism: It uses task-specific instructions. For retrieval, the query is prefixed with "Instruct: Given a web search query, retrieve relevant passages that answer the query." This asymmetry aligns the vector space more effectively for Q&A tasks.   

Pros: Top-tier performance on the MTEB leaderboard. Supports 100+ languages. Handles the semantic gap between questions and answers better than BERT.

Cons: Heavier computational load (Large model).

2.2.3. Candidate 3: BAAI BGE-M3 (BAAI/bge-m3)
A newer entrant focusing on Multi-Linguality, Multi-Functionality, and Multi-Granularity.

Pros: Supports dense retrieval, sparse retrieval (like BM25), and multi-vector retrieval (ColBERT-style) simultaneously. Extremely robust.

Cons: Significant complexity in implementation. The multi-vector approach requires specialized vector stores and retrieval logic which may be overkill for a textbook RAG application.   

2.3. Strategic Selection: The E5 Advantage
For Scorates, intfloat/multilingual-e5-large-instruct is selected as the primary embedding engine.

Justification:

Instruction Tuning: The ability to differentiate between the role of the text (query vs. document) is critical for a Socratic tutor. The system needs to match a student's confused question not just to keywords, but to the explanatory sections of the text. E5's instruction tuning excels here.

Multilingual robustness: While st-greek-media-bert is Greek-specific, E5's training on a larger, more diverse corpus (including Wikipedia) makes it more robust for academic and formal Greek found in textbooks, compared to the media/social-media heavy training of the specialized Greek BERT.

Dimensionality: The 1024 dimensions allow for a richer semantic space than the 768 dimensions of the base BERT model, necessary to capture the nuanced conceptual relationships in subjects like History or Philosophy.

2.4. Text Extraction and Preprocessing
To feed the embedding model, we must extract text from the PDF textbooks with high fidelity.

Tool Selection: PyMuPDF (fitz) PyMuPDF is consistently benchmarked as the fastest and most accurate PDF extraction library for Python. Crucially for Greek, it handles standard CMap (Character Map) extractions effectively, mitigating encoding errors better than pypdf or pdfminer.six.   

Preprocessing Pipeline:

Extraction: Extract text blocks using page.get_text("dict") to preserve layout information (headers vs. body text).

Normalization: Apply Unicode Normalization Form C (NFC). This is non-negotiable for Greek. It ensures that the character 'ά' is stored as a single code point (U+03AC) rather than a sequence of 'α' + 'tone' (U+03B1 + U+0301). This standardization is required for the embedding model's tokenizer to recognize words correctly.   

Cleaning: Remove headers, footers, and page numbers which are artifacts of the PDF and break the semantic flow of the text.

Chunking Strategy: We will employ a Recursive Character Text Splitter with specific parameters for Greek:

Chunk Size: 512 tokens (approx. 1500-2000 characters). This aligns with the input limit of the E5 model.

Overlap: 10% (50 tokens). This preserves context across boundaries, ensuring that a sentence split in the middle is still semantically complete in at least one chunk.

Separators: ["\n\n", "\n", ". ", " ", ""]. Splitting by paragraph (\n\n) is prioritized to keep ideas intact.

3. Architectural Design: The Clean Approach
To ensure the longevity, testability, and maintainability of Scorates, we will implement Clean Architecture (often referred to as Hexagonal Architecture or Ports and Adapters). This approach segregates the software into layers, with dependencies pointing strictly inwards. The core business logic (Pedagogy) depends on nothing; the external tools (databases, web frameworks) depend on the core logic.

3.1. High-Level Architectural Layers
The application is structured into four concentric circles:

3.1.1. Domain Layer (The Core)
This is the innermost layer. It contains the Enterprise Business Rules. In the context of Scorates, these are the fundamental concepts of the application that would exist even if it were a paper-based system.

Entities:

Student: The user profile.

StudyMaterial: The textbook or PDF being analyzed.

Conversation: The history of the dialogue.

SocraticQuery: A specific question formulated by the system.

Interfaces (Ports): This layer defines how it expects to interact with the outside world but implies no implementation. It defines interfaces for VectorStore, DocumentLoader, and LLMProvider.

Dependencies: None. Pure Python.

3.1.2. Application Layer (The Use Cases)
This layer contains the Application Business Rules. It orchestrates the flow of data to and from the Domain entities to achieve specific user goals.

Use Cases (Interactors):

IngestTextbook: Orchestrates loading a PDF, cleaning it, chunking it, and sending it to storage.

ConductSocraticDialogue: The core logic that receives a student message, retrieves context, and calls the LLM to generate a question.

AssessStudentUnderstanding: A background process that analyzes the conversation history to update the student's progress model.

Dependencies: Depends only on the Domain Layer.

3.1.3. Infrastructure Layer (The Adapters)
This layer adapts the interface of external frameworks and tools to the interfaces defined in the Domain Layer.

Implementations:

ChromaDBAdapter: Implements VectorStore using ChromaDB.

LangChainLLMAdapter: Implements LLMProvider using LangChain's OpenAI wrappers.

PyMuPDFLoader: Implements DocumentLoader.

Dependencies: Depends on the Application Layer interfaces and external libraries (LangChain, Chroma, OpenAI SDK).

3.1.4. Presentation Layer (The UI/API)
The outermost layer that interacts with the user.

Components:

FastAPI Controllers: REST endpoints for mobile or web clients.

Chainlit App: The specific chat interface logic.

Dependencies: Depends on the Application Layer to execute actions.

3.2. Technology Stack Justification
Component	Technology	Rationale
Language	Python 3.11+	
The lingua franca of AI. Version 3.11+ offers significant speed improvements and better async support, critical for high-concurrency chat applications.

Web Framework	FastAPI	
Selected for its asynchronous core, automatic OpenAPI (Swagger) documentation, and Pydantic integration. It allows for high-performance handling of concurrent chat sessions.

Orchestration	LangChain	
Provides a standardized interface for LLM interaction. Its "Runnables" protocol (LCEL) allows for clean composition of RAG chains. While LangChain can be complex, wrapping it in the Infrastructure layer keeps the core clean.

Vector Database	ChromaDB	
An open-source, AI-native vector database. It is easy to run locally via Docker, integrates natively with LangChain, and supports the metadata filtering required to restrict searches to specific textbooks.

LLM	GPT-4o / Llama 3	
GPT-4o is the current gold standard for complex reasoning (required for Maieutics). Llama 3 (via Groq or local Ollama) serves as a cost-effective open-source alternative for development.

Frontend	Chainlit	
Chosen over Streamlit for this specific use case. Chainlit is purpose-built for chat interfaces. It supports "Chain of Thought" visualization, streaming responses, and element attachments (images/PDFs) out of the box. Streamlit requires more boilerplate for state management in chat applications.

  
4. Deep Dive: The RAG Socratic Pipeline
The standard RAG pipeline (Retrieve -> Augment -> Generate) must be modified to support the Socratic method. The goal is not to answer the question, but to find the information required to answer the question and then use that information to formulate a new question.

4.1. The Pipeline Steps
Query Analysis & Reformulation:

The student's input (e.g., "Why did it happen?") is often context-dependent.

Step: Use a lightweight LLM call to rewrite the query into a standalone statement based on chat history (e.g., "Why did the French Revolution start?").

Language: This must happen in Greek.

Context Retrieval (Hybrid Search):

Dense Retrieval: Embed the reformulated Greek query using multilingual-e5-large-instruct. Search the Vector Store.

Keyword Retrieval (BM25): Greeks words often have specific technical meanings in textbooks. A keyword search ensures that specific terms (e.g., "Φιλική Εταιρεία") are strictly matched.

Ensemble: Combine results using Reciprocal Rank Fusion (RRF) to get the best of both worlds.   

Context Reranking:

The top 10-20 results from the retrieval step are often noisy.

Step: Pass the query and the 20 documents to a Cross-Encoder (e.g., BAAI/bge-reranker-v2-m3). This model scores every document pair for relevance.

Output: The top 3 most relevant text chunks. This high precision is vital; if the context is wrong, the Socratic question will be irrelevant.

Pedagogical System Prompting (The "Maieutic Core"):

The retrieved context and the student's query are fed into the LLM.

The System Prompt (detailed in Section 6) enforces the Socratic constraints.

Logic:

Does the student's answer show understanding? -> Validate and move to next topic.

Does the student's answer show misconception? -> Ask a question that reveals the contradiction.

Is the student asking for the answer? -> Provide a hint or a question that leads to the first step of the solution.

5. Technical Implementation & Codebase Skeleton
This section provides the concrete implementation details, adhering to the Clean Architecture defined above.

5.1. Project Directory Structure
A modular structure allows teams to work on different parts (e.g., prompt engineering vs. database optimization) without stepping on toes.   

scorates/ ├── app/ │ ├── init.py │ ├── domain/ # INNERMOST LAYER: Pure Business Logic │ │ ├── init.py │ │ ├── models.py # Pydantic entities (StudyMaterial, Message) │ │ ├── ports.py # Interfaces (ABCs) │ │ └── exceptions.py # Domain-specific errors │ ├── application/ # USE CASE LAYER: Orchestration │ │ ├── init.py │ │ ├── services/ │ │ │ ├── ingestion_service.py # PDF processing logic │ │ │ ├── chat_service.py # Socratic logic orchestration │ │ │ └── session_service.py # State management │ │ └── dtos.py # Data Transfer Objects │ ├── infrastructure/ # ADAPTER LAYER: External Tools │ │ ├── init.py │ │ ├── db/ │ │ │ ├── chroma_adapter.py # Vector Store implementation │ │ │ └── sql_adapter.py # History storage │ │ ├── llm/ │ │ │ ├── langchain_adapter.py # LangChain wrapper │ │ │ └── prompts.py # Greek System Prompts │ │ └── pdf/ │ │ └── pymupdf_adapter.py # PDF extraction │ ├── presentation/ # OUTERMOST LAYER: UI/API │ │ ├── api/ │ │ │ ├── main.py # FastAPI entry point │ │ │ └── routes.py # API endpoints │ │ └── ui/ │ │ └── chainlit_app.py # Chat Interface │ └── config.py # Environment configuration ├── data/ # Local storage (Docker volumes) ├── docker/ │ ├── Dockerfile │ └── docker-compose.yml ├── tests/ # Pytest suite ├── pyproject.toml # Dependency management (Poetry/UV) └── README.md

5.2. Domain Layer Implementation
app/domain/models.py Here we define the core data structures using Pydantic. Note the use of UUIDs for robust identification.

Python
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
app/domain/ports.py These Abstract Base Classes (ABCs) enforce the Dependency Inversion Principle. The Application layer will type-hint against these classes, not the specific implementations.

Python
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
5.3. Infrastructure Layer Implementation
app/infrastructure/pdf/pymupdf_adapter.py This adapter encapsulates the complexity of fitz (PyMuPDF). It normalizes the Greek text during extraction.

Python
import fitz  # PyMuPDF
import unicodedata
from app.domain.ports import DocumentLoaderPort
from langchain.docstore.document import Document
from typing import List

class PyMuPDFLoaderAdapter(DocumentLoaderPort):
    def load_pdf(self, file_path: str) -> List:
        doc = fitz.open(file_path)
        documents =
        
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
app/infrastructure/llm/prompts.py This file contains the Pedagogical Engine's Brain. The prompt is written in Greek to ensure the LLM reasons in the target language.

Python
# System Prompt for Socratic Tutor in Greek
# Incorporates: Role definition, Constraints, Methodology, and Tone.

SOCRATIC_SYSTEM_PROMPT_EL = """
Είσαι ο Σωκράτης (Scorates), ένας ψηφιακός μέντορας για μαθητές Γυμνασίου και Λυκείου στην Ελλάδα.
Η αποστολή σου ΔΕΝ είναι να δίνεις έτοιμες απαντήσεις, αλλά να καλλιεργείς την κριτική σκέψη μέσω της Μαιευτικής μεθόδου.

ΟΔΗΓΙΕΣ ΠΑΙΔΑΓΩΓΙΚΗΣ (PROTOCOL):
1. **Απαγόρευση Απευθείας Απάντησης:** Αν ο μαθητής ρωτήσει κάτι που υπάρχει στο κείμενο, ΜΗΝ το απαντήσεις. Αντίθετα, κάνε μια ερώτηση που θα τον οδηγήσει να το βρει μόνος του στο κείμενο.
2. **Χρήση Context:** Όλες οι ερωτήσεις σου πρέπει να βασίζονται ΑΠΟΚΛΕΙΣΤΙΚΑ στο παρεχόμενο κείμενο (Context). Μην χρησιμοποιείς εξωτερικές γνώσεις.
3. **Βήμα-Βήμα (Scaffolding):** Αν η ερώτηση είναι δύσκολη, σπάσε την σε απλούστερα υπο-ερωτήματα.
4. **Διαχείριση Λάθους:** Αν ο μαθητής κάνει λάθος, μην τον διορθώσεις άμεσα. Ρώτησε: "Πώς κατέληξες σε αυτό το συμπέρασμα;" ή δώσε ένα αντι-παράδειγμα από το κείμενο.
5. **Ύφος:** Φιλικό, ενθαρρυντικό, αλλά ακαδημαϊκό. Χρησιμοποίησε Δημοτική γλώσσα.

ΔΟΜΗ ΑΠΑΝΤΗΣΗΣ:
- Ξεκίνα με μια επιβεβαίωση ή μια γέφυρα με τα λεγόμενα του μαθητή.
- Διατύπωσε την Σωκρατική ερώτηση καθαρά.
- Αν χρειάζεται, δώσε μια μικρή βοήθεια (hint) που παραπέμπει σε συγκεκριμένη ενότητα του κειμένου (π.χ. "Κοίταξε την παράγραφο για τις αιτίες...").

CONTEXT (Από το βιβλίο):
{context}

ΙΣΤΟΡΙΚΟ ΣΥΝΟΜΙΛΙΑΣ:
{chat_history}

ΜΑΘΗΤΗΣ: {question}
"""
5.4. Application Layer Implementation
app/application/services/chat_service.py This service ties the components together. It implements the RAG pipeline logic.

Python
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
5.5. Presentation Layer Implementation
We leverage Chainlit for the frontend. Its pythonic nature allows us to integrate it deeply with our backend logic without writing React/JS code. It supports streaming, which is essential for user engagement.

app/presentation/ui/chainlit_app.py

Python
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
6. Infrastructure & Deployment (Docker Strategy)
To ensure the application is portable and production-ready, we utilize Docker. Specifically, we employ a Multi-Stage Build to keep the final image size small and secure. The Python image includes build tools (GCC) in the first stage to compile libraries, but the final stage only contains the runtime artifacts.   

6.1. Optimized Dockerfile
Dockerfile
# STAGE 1: Builder
# Use a full python image to build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies required for building python packages (e.g. PyMuPDF, Chroma)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python Dependency Manager (using pip for simplicity here, but Poetry is recommended)
COPY requirements.txt.
# Install into a local directory that we can copy later
RUN pip install --user --no-cache-dir -r requirements.txt

# STAGE 2: Runtime
# Use a slim image for production
FROM python:3.11-slim as runtime

WORKDIR /app

# Create a non-root user for security
RUN useradd -m scorates
USER scorates

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/scorates/.local

# Update PATH
ENV PATH=/home/scorates/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy Application Code
COPY --chown=scorates:scorates..

# Expose the port
EXPOSE 8000

# Start Command using Uvicorn (Production Server)
# We point to the FastAPI app wrapper that mounts Chainlit
CMD ["uvicorn", "app.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
6.2. Docker Compose Configuration
This configuration spins up the API and the Vector Database service.

YAML
version: '3.8'

services:
  scorates-app:
    build:.
    container_name: scorates_app
    ports:
      - "8000:8000"
    volumes:
      -./app:/app/app  # Hot reload for development
      -./data:/app/data
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHROMA_DB_HOST=chroma-db
      - CHROMA_DB_PORT=8000
    depends_on:
      - chroma-db
    networks:
      - scorates-net

  chroma-db:
    image: chromadb/chroma:latest
    container_name: scorates_vector_db
    ports:
      - "8001:8000"
    volumes:
      - chroma-data:/chroma/chroma
    networks:
      - scorates-net

volumes:
  chroma-data:

networks:
  scorates-net:
    driver: bridge
7. Deep Insights and Strategic Recommendations
7.1. The "Pedagogical Hallucination" Risk
A unique risk in Socratic AI is not just factual hallucination (making up dates) but pedagogical hallucination. This occurs when the AI asks a question that is conceptually sound but cannot be answered using the provided text. For example, asking a student to "Compare this to the American Revolution" when the textbook only covers the French Revolution.

Implication: This breaks the trust in the tool as a "study buddy" for a specific exam.

Mitigation Strategy: Implement a "Solvability Check" layer. Before displaying the question to the user, the LLM should be prompted internally: "Does the retrieved text contain sufficient information to answer the question you just generated? If no, generate a simpler question based ONLY on the text."

7.2. Tokenization Economics for Greek
Greek text is token-dense. A standard OpenAI token represents ~0.75 English words but often only ~0.3 Greek words.

Implication: API costs for Greek applications will be approximately 2-3x higher than equivalent English applications. Context windows fill up faster.

Recommendation: Aggressive reranking is not just for quality; it is a cost-saving measure. By narrowing the context from 10 chunks to 3 highly relevant chunks via a local Cross-Encoder (which is free to run), we significantly reduce the token load sent to the paid LLM API.

7.3. Asymmetric Language Proficiency
While GPT-4o is excellent at Greek, smaller open-source models (like Llama 3 8B) often have "asymmetric proficiency." They understand Greek inputs well (reading) but may struggle to generate high-quality, nuanced Greek Socratic questions (writing), often reverting to English grammatical structures or hallucinating terminology.

Recommendation: For the MVP, rely on GPT-4o. If moving to open-source for cost reduction, fine-tuning a Llama-3 model on a dataset of Greek educational dialogues is mandatory.

8. Conclusion
Scorates represents a sophisticated fusion of classical educational philosophy and modern software architecture. By rigorously defining the "Socratic" requirements and mapping them to specific technical decisions—such as the use of multilingual-e5-large-instruct for embeddings, FastAPI/Chainlit for the stack, and Clean Architecture for code organization—this specification provides a roadmap for building a tool that truly serves the Greek student.

The system avoids the trap of becoming a homework-solving machine. Instead, it positions itself as a tireless tutor, endlessly patient, culturally aware, and strictly guided by the curriculum. The path forward involves prototyping the RAG pipeline using the provided Python skeleton, validating the retrieval quality with real Greek textbooks, and iterating on the system prompt to perfect the delicate art of the Socratic question.

Citations
   

End of Report


sehd.ucdenver.edu
AI Prompting – Socratic Method – SEHD Impact
Opens in a new window

solve.mit.edu
Socratic Mind - MIT Solve
Opens in a new window

diplomacy.edu
What can Socrates teach us about AI and prompting? - Diplo Foundation
Opens in a new window

growthengineering.co.uk
The Socratic Method: Your Complete Guide - Growth Engineering
Opens in a new window

ceur-ws.org
Generative AI for Teaching Latin and Greek in High School - CEUR-WS
Opens in a new window

talkpal.ai
The Beginner's Blueprint to Learning Greek with ChatGPT - Talkpal
Opens in a new window

reddit.com
Εσυ/εσείς use : r/GREEK - Reddit
Opens in a new window

jktauber.com
Python, Unicode and Ancient Greek - J. K. Tauber
Opens in a new window

stackoverflow.com
Searching for greek characters within a PDF document - Stack Overflow
Opens in a new window

modal.com
Top embedding models on the MTEB leaderboard | Modal Blog
Opens in a new window

reddit.com
Open-source embedding models: which one's the best? : r/Rag - Reddit
Opens in a new window

dataloop.ai
Multilingual E5 Base · Models - Dataloop
Opens in a new window

huggingface.co
dimitriz/st-greek-media-bert-base-uncased - Hugging Face
Opens in a new window

huggingface.co
BERT - Hugging Face
Opens in a new window

dataloop.ai
St Greek Media Bert Base Uncased · Models - Dataloop
Opens in a new window

huggingface.co
intfloat/multilingual-e5-large - Hugging Face
Opens in a new window

huggingface.co
intfloat/multilingual-e5-large-instruct - Hugging Face
Opens in a new window

pymupdf.readthedocs.io
Features Comparison - PyMuPDF documentation
Opens in a new window

reddit.com
Which is faster at extracting text from a PDF: PyMuPDF or PyPDF2? : r/learnpython - Reddit
Opens in a new window

medium.com
Comparing 4 methods for pdf text extraction in python | by Jeanna Schoonmaker | Social Impact Analytics | Medium
Opens in a new window

datacamp.com
Building a RAG System with LangChain and FastAPI: From Development to Production
Opens in a new window

stochasticcoder.com
LangChain RAG with React, FastAPI, Cosmos DB Vector: Part 2 - Stochastic Coder
Opens in a new window

github.com
Harmeet10000/AgentNexus-LangChain-FastAPI - GitHub
Opens in a new window

medium.com
From Zero to RAG Chatbot in 10 Minutes (LangChain + Qdrant + Mistral + FastAPI + Airflow) | by Neelamyadav | Medium
Opens in a new window

blog.futuresmart.ai
Building a Production-Ready RAG Chatbot with FastAPI and LangChain
Opens in a new window

reddit.com
Open-source embedding models: which one to use? : r/LocalLLaMA - Reddit
Opens in a new window

app.daily.dev
Rapid Prototyping of Chatbots with Streamlit and Chainlit - daily.dev
Opens in a new window

youtube.com
Streamlit vs Chainlit: Which is Better for AI Apps? | Beginners Guide - YouTube
Opens in a new window

slashdot.org
Compare Chainlit vs. Streamlit in 2025 - Slashdot
Opens in a new window

medium.com
Building Interactive AI Applications with Chainlit: An Open-Source Framework for LLM Interfaces | by catlin | Medium
Opens in a new window

sbert.net
Semantic Search - Sentence Transformers documentation
Opens in a new window

youtube.com
GMR 229: Semantic Search using Sentence Transformers - YouTube
Opens in a new window

github.com
wassim249/fastapi-langgraph-agent-production-ready-template - GitHub
Opens in a new window

testdriven.io
Docker Best Practices for Python Developers - TestDriven.io
Opens in a new window

collabnix.com
Docker Multi-Stage Builds for Python Developers: A Complete Guide - Collabnix
Opens in a new window

chemrxiv.org
The Hitchhiker's Guide to Socratic Methods in Prompting Large Language Models for Chemistry Applications - ChemRxiv
Opens in a new window

docs.chainlit.io
FastAPI - Chainlit