# Scorates: A Socratic AI Tutor for Greek Secondary Education

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![LangChain](https://img.shields.io/badge/LangChain-v0.1-orange)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **"Education is the kindling of a flame, not the filling of a vessel."** — Socrates

## Executive Summary

**Scorates** is a study support application specifically engineered for the Greek secondary education system (Gymnasio and Lykeio). Unlike standard "answer engines" (like ChatGPT), Scorates is a **Digital Maieutic Engine**.

Deriving its core philosophy from the Socratic method (*maieutikos*), the system is designed to "birth" knowledge from the student rather than delivering it. By integrating advanced **Retrieval-Augmented Generation (RAG)** architectures with a strictly constrained pedagogical persona, Scorates transforms the interaction from passive consumption to active inquiry.

This repository contains the comprehensive technical and pedagogical implementation, prioritizing **pedagogical integrity** over speed, **architectural cleanliness** over rapid prototyping, and **linguistic precision** for Modern Greek.

---

## Table of Contents
1. [Pedagogical Framework](#1-pedagogical-framework-the-digital-socratic-method)
2. [NLP Strategy for Modern Greek](#2-natural-language-processing-strategy-for-modern-greek)
3. [Architecture](#3-architectural-design-the-clean-approach)
4. [RAG Pipeline](#4-deep-dive-the-rag-socratic-pipeline)
5. [Implementation](#5-technical-implementation--codebase-skeleton)
6. [Deployment](#6-infrastructure--deployment-docker-strategy)
7. [Insights & Risks](#7-deep-insights-and-strategic-recommendations)

---

## 1. Pedagogical Framework: The Digital Socratic Method

### 1.1 The Crisis of the "Answer Engine"
Standard LLMs foster surface-level engagement by providing direct answers, bypassing the "desirable difficulty" required for deep learning. Scorates refuses to be an encyclopedia and insists on being a dialectic partner.

### 1.2 Maieutics: The Core Mechanics
The system emulates three cognitive movements:
* **Elenchus (Refutation):** Identifying contradictions in the student's premise.
* **Aporia (Perplexity):** Guiding the student to a state of curiosity.
* **Maieutics (Birthing):** Assisting the student in constructing the answer based on their reasoning.

### 1.3 The "Guide-on-the-Side" Constraints
* **Prohibition of Direct Answers:** Never solve the problem if the text contains the solution.
* **Contextual Anchoring:** All questions must be solvable using the uploaded PDF (Ministry of Education textbooks).
* **Incremental Complexity:** Mirror Bloom's Taxonomy (Remembering $\to$ Evaluating).

---

## 2. Natural Language Processing Strategy for Modern Greek

### 2.1 Linguistic Challenges
* **Morphological Richness:** Handling Greek inflections (e.g., *μαθητής, μαθητές, μαθητή*).
* **Encoding:** Normalizing Unicode to prevent "mojibake" (NFC normalization).
* **Tokenization:** Optimizing for the high token cost of Greek text.

### 2.2 Selected Embedding Model
We utilize **`intfloat/multilingual-e5-large-instruct`**.
* **Why?** It supports "Instruction Tuning," allowing the model to differentiate between a *query* and a *document*. It offers superior robustness for academic Greek compared to media-trained BERT models.

### 2.3 Text Preprocessing
* **Tool:** PyMuPDF (`fitz`) for CMap accuracy.
* **Normalization:** Unicode Normalization Form C (NFC) is mandatory.
* **Chunking:** Recursive Character Split (512 tokens, 10% overlap).

---

## 3. Architectural Design: The Clean Approach

We implement **Clean Architecture** (Ports and Adapters) to segregate business logic from external tools.

### 3.1 Tech Stack
| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Async support for high concurrency. |
| **Web Framework** | FastAPI | High-performance, automatic OpenAPI documentation. |
| **Orchestration** | LangChain | LCEL for composable RAG chains. |
| **Vector DB** | ChromaDB | Open-source, AI-native, easy Docker integration. |
| **LLM** | GPT-4o / Llama 3 | GPT-4o for complex reasoning; Llama 3 for local dev. |
| **Frontend** | Chainlit | Purpose-built for Chat UIs with "Chain of Thought" views. |

---

## 4. Deep Dive: The RAG Socratic Pipeline

1.  **Query Reformulation:** Rewrite student input into a standalone Greek statement.
2.  **Hybrid Retrieval:**
    * *Dense:* Vector search via E5 embeddings.
    * *Sparse:* BM25 for specific Greek terminology (e.g., "Φιλική Εταιρεία").
3.  **Reranking:** Use a Cross-Encoder to select the top 3 most relevant chunks.
4.  **Pedagogical Prompting:** The "Maieutic Core" analyzes if the student needs a hint, a counter-example, or a follow-up question.

---

## 5. Technical Implementation & Codebase Skeleton

### 5.1 Directory Structure
```text
scorates/
├── app/
│   ├── domain/           # INNERMOST: Pure Business Logic
│   ├── application/      # USE CASES: Orchestration
│   ├── infrastructure/   # ADAPTERS: External Tools (DB, LLM)
│   └── presentation/     # OUTERMOST: API & UI
├── data/                 # Local storage
├── docker/               # Docker configs
└── pyproject.toml        # Dependencies