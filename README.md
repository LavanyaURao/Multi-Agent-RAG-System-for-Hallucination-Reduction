# Multi-Agent RAG System for Hallucination Reduction

## Problem Statement
Large Language Models continue to produce inaccurate responses due to hallucinations, particularly during complex reasoning, even when supported by Retrieval-Augmented Generation. While existing methods improve factual grounding, they still need better query planning, evidence filtering, and retry logic when the first answer is weak.

## Abstract
This project implements a lightweight multi-agent RAG framework in which specialized agents handle query understanding, multi-retriever evidence gathering, context verification, answer generation, critique, and final judgment. The system now adds a context gate before generation and a controlled feedback loop after critique to reduce unsupported claims more reliably than a single-pass generator.

## Models Used

| Component | Model | Provider |
|---|---|---|
| Embedding / Retrieval | `all-MiniLM-L6-v2` | HuggingFace (local) |
| Query Understanding Agent | `llama-3.1-8b-instant` | Groq API |
| Context Verifier Agent | `llama-3.1-8b-instant` | Groq API |
| Generator Agent | `llama-3.3-70b-versatile` | Groq API |
| Critic Agent | `llama-3.1-8b-instant` | Groq API |
| Judge Agent | `llama-3.1-8b-instant` | Groq API |

## Project Structure

```text
GenAI/
|-- agents/
|   |-- context_verifier.py      # Context Verifier Agent
|   |-- critic.py                # Critic Agent
|   |-- generator.py             # Generator Agent
|   |-- judge.py                 # Judge Agent
|   |-- query_understanding.py   # Query Understanding Agent
|   `-- retriever.py             # Multi-Retriever Agent
|-- data/
|   `-- knowledge_base.txt       # Domain knowledge
|-- database/
|   `-- vector_store.py          # FAISS + sentence-transformers
|-- pipeline/
|   `-- multi_agent_pipeline.py  # Orchestrates all agents
|-- app.py                       # Streamlit app
|-- main.py                      # CLI entry point
|-- requirements.txt
`-- README.md
```

## Setup & Run

### 1. Configure `.env`
```bash
GROQ_API_KEY=your_key_here
```

### 2. Create virtual environment
```bash
python -m venv venv
```

### 3. Activate environment
```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run
```bash
python main.py
```

Results are saved to `results.json`.

## Updated Pipeline Flow

```text
User Query
    |
    v
[Query Understanding Agent]
    |
    v
[Multi-Retriever System]
  semantic FAISS + keyword overlap retrieval
    |
    v
[Context Verifier]
    |
    v
[Generator]
    |
    v
[Critic]
    |
    v
[Feedback Loop]
  optional one-pass retrieve/generate retry
    |
    v
[Judge]
    |
    v
Final Answer
```

## Why These Additions Help

- `Query Understanding Agent` improves retrieval quality by separating user phrasing from retrieval phrasing.
- `Multi-Retriever System` reduces blind spots from relying on only one retriever.
- `Context Verifier` filters noisy evidence before generation.
- `Feedback Loop` gives the system one bounded retry when critique detects issues.

## LangGraph Recommendation

LangGraph is a good addition if you want this project to become more stateful and branch-heavy, especially if you plan to add:

- multiple retry policies
- conditional routing between different retrievers or tools
- checkpoints or resumable execution
- tracing of node-level state transitions
- human review steps

For the current codebase, LangGraph is useful but not required. The workflow is still simple enough to keep in plain Python without losing clarity. If your next step is just one feedback loop plus a few agents, stay with the current structure. If you expect rapid growth in branching and orchestration logic, then migrating to LangGraph is a good idea.
