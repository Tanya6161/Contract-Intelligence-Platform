# 📑 Contract Intelligence Platform

AI-powered contract analysis platform built using CrewAI, Gemini, LangChain, ChromaDB, and Streamlit.

The application enables users to upload contract PDFs and receive structured insights including key clauses, risk indicators, obligations, and negotiation recommendations through a Retrieval-Augmented Generation (RAG) pipeline and multi-agent AI architecture.

---

## 🚀 Features

- Contract PDF Upload
- Multi-Agent Analysis using CrewAI
- Semantic Clause Retrieval
- Risk Identification
- Negotiation Recommendations
- Retrieval-Augmented Generation (RAG)
- Vector Search with ChromaDB
- HuggingFace Embeddings
- Interactive Streamlit Dashboard
- Structured JSON Output

---

## 🏗 Architecture

```text
PDF Upload
     │
     ▼
PyPDFLoader
     │
     ▼
RecursiveCharacterTextSplitter
     │
     ▼
HuggingFace Embeddings
     │
     ▼
ChromaDB Vector Store
     │
     ▼
Semantic Search
     │
     ▼
CrewAI Agents
     │
     ▼
Gemini LLM
     │
     ▼
Contract Analysis Results
```

---

## 🤖 AI Agents

### Ingestion Agent
- Loads PDF contracts
- Extracts text
- Chunks documents
- Creates embeddings

### Retrieval Agent
- Performs semantic search
- Retrieves relevant clauses
- Identifies important sections

### Analyst Agent
- Detects risks
- Summarizes obligations
- Generates negotiation suggestions

---

## 🧠 AI Concepts Used

### Beginner
- Large Language Models (LLMs)
- Prompt Engineering
- Generative AI

### Intermediate
- Embeddings
- Semantic Search
- Vector Databases
- Multi-Agent Systems

### Advanced
- Retrieval-Augmented Generation (RAG)
- Agent Orchestration
- Context-Aware Retrieval
- LLM Grounding

---

## 🛠 Tech Stack

- Python
- Streamlit
- CrewAI
- Gemini
- LangChain
- ChromaDB
- HuggingFace Embeddings
- PyPDF
- Sentence Transformers

---

## 📸 Application Preview

### Dashboard

![Dashboard](screenshots/home_page.png)

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/contract-intelligence-platform.git
```

Move into project folder:

```bash
cd contract-intelligence-platform
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create environment file:

```env
GOOGLE_API_KEY=your_api_key
```

Run application:

```bash
streamlit run contract_analysis_app.py
```

---

## 🎯 Future Enhancements

- Clause Comparison Engine
- Contract Version Tracking
- Compliance Scoring
- Multi-Contract Analysis
- Legal Knowledge Graph
- Dashboard Analytics

---

## ⚠ Disclaimer

This project provides AI-assisted contract analysis and should not replace professional legal review.
