# The Complete Guide to Generative AI

**From Zero to Building Real-World Applications**

---
 
## Who Is This Guide For?

This guide is designed for college students, aspiring developers, and anyone who wants to understand Generative AI from the ground up and build real projects. No prior AI experience is required — just basic Python knowledge.

By the end, you will be able to:
- Understand all core GenAI concepts (LLMs, Transformers, Embeddings, RAG)
- Build real-world applications using LangChain and RAG
- Create strong portfolio projects for placements and job interviews
- Confidently explain everything in technical interviews

---

## Table of Contents

| # | Chapter | What You'll Learn |
|---|---------|-------------------|
| 0 | [Fast-Track Plan](00-fast-track-plan.md) | 10-15 day accelerated learning roadmap |
| 1 | [Foundations of AI & Generative AI](01-foundations.md) | What AI is, types of AI, how GenAI differs |
| 2 | [Introduction to LLMs](02-intro-to-llms.md) | How large language models work, key models, tokenization |
| 3 | [Transformers](03-transformers.md) | The architecture behind modern AI, attention mechanism |
| 4 | [Embeddings & Vector Databases](04-embeddings-vectordbs.md) | How AI understands meaning, similarity search |
| 5 | [Prompt Engineering](05-prompt-engineering.md) | Writing effective prompts, techniques, beginner projects |
| 6 | [LangChain & LLM App Development](06-langchain.md) | Building apps with LangChain, chains, memory, intermediate projects |
| 7 | [Retrieval-Augmented Generation](07-rag.md) | Connecting LLMs to your data, advanced projects |
| 8 | [Building End-to-End GenAI Apps](08-end-to-end-apps.md) | Full-stack GenAI applications, deployment |
| 9 | [Advanced Topics](09-advanced-topics.md) | AI Agents, fine-tuning, multimodal AI |
| 10 | [Final Projects & Interview Prep](10-projects-interview.md) | Resume-level projects, interview questions & answers |

---

## How to Use This Guide

### Reading Path
- **Complete beginner?** Start from Chapter 1 and go sequentially.
- **Know some AI basics?** Skip to Chapter 3 (Transformers) or Chapter 5 (Prompt Engineering).
- **Want to build NOW?** Jump to Chapter 6 (LangChain) — refer back as needed.
- **Short on time?** Follow the [Fast-Track Plan](00-fast-track-plan.md).

### Each Chapter Contains
1. **Concept Explanation** — Beginner-friendly with analogies
2. **Detailed Notes** — Book-style, step-by-step breakdown
3. **Visual Explanation** — Diagrams described in words
4. **YouTube Resources** — Recommended channels and search terms
5. **Official Documentation** — What to read and why
6. **Code Examples** — Runnable Python snippets (OpenAI + Claude)
7. **Mini Practice Tasks** — Small exercises to test understanding
8. **Quick Revision Summary** — Key points for quick review
9. **Common Mistakes** — What beginners get wrong

### Projects Roadmap

| Level | After Chapter | Projects |
|-------|--------------|----------|
| Beginner | Ch 5 - Prompt Engineering | Chatbot, Text Generator, Email Writer |
| Intermediate | Ch 6 - LangChain | Quiz Generator, Document Q&A, Chatbot with Memory |
| Advanced | Ch 7 - RAG | PDF Chatbot, Learning Assistant, Knowledge Base |
| Resume-Level | Ch 10 - Final | 3 industry-grade projects with full architecture |

---

## Prerequisites

- **Python basics** (variables, functions, loops, pip)
- **A code editor** (VS Code recommended)
- **API keys** from OpenAI and/or Anthropic (free tiers available)
- **Curiosity and willingness to experiment**

## Setup

```bash
# Create a virtual environment
python -m venv genai-env
source genai-env/bin/activate  # Mac/Linux
# genai-env\Scripts\activate   # Windows

# Install core packages
pip install openai anthropic langchain langchain-openai langchain-anthropic
pip install chromadb tiktoken python-dotenv streamlit

# Set up API keys (create a .env file)
echo "OPENAI_API_KEY=your-key-here" > .env
echo "ANTHROPIC_API_KEY=your-key-here" >> .env
```

---

*Let's begin your GenAI journey.*
