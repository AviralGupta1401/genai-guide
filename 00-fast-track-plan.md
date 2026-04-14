# 10-15 Day Fast-Track Plan for Generative AI

> **Who is this for?** College students or early-career developers who want to go from zero to building real GenAI applications in about two weeks. No PhD required.

> **How to use this plan:** Each day has a **Morning (Learn)** and **Afternoon (Build)** block. Aim for 4-6 focused hours per day. If you only have 2-3 hours, follow the "Short on Time?" notes to cut the right corners.

---

## Before You Start

### What to Prioritize

These are the things that will get you hired and help you build real products:

- **Prompt engineering** -- this is the single highest-ROI skill in GenAI right now
- **Building things** -- recruiters care about projects, not theory
- **RAG (Retrieval-Augmented Generation)** -- the most common production pattern
- **API usage** -- OpenAI, Anthropic, and open-source model APIs
- **LangChain / LlamaIndex basics** -- the dominant orchestration frameworks
- **Vector databases** -- you will use these in almost every GenAI app

### What to Skip (For Now)

Do not spend time on these during your fast-track sprint. You can always come back later:

- **Heavy linear algebra and calculus** -- you need intuition, not proofs
- **History of AI from the 1950s** -- interesting but won't help you build
- **Training models from scratch** -- you will use pre-trained models and fine-tuning
- **Reading full research papers end-to-end** -- read summaries and blog posts instead
- **Optimizing GPU clusters** -- that's an infrastructure job, not your focus right now
- **Every single LangChain module** -- learn the 20% you'll use 80% of the time

---

## The Daily Plan

---

### Days 1-2: Foundations + LLM Basics

**Reference:** [Chapter 1 -- Foundations](01-foundations.md) | [Chapter 2 -- LLM Basics](02-llm-basics.md)

#### Day 1

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | What is Generative AI? | Read Ch 1. Understand the difference between traditional ML and generative models. Learn what LLMs are, how they generate text, and why they matter. |
| **Afternoon (Build)** | Set up your environment | Install Python 3.10+, set up a virtual environment, get API keys (OpenAI and/or Anthropic). Make your first API call. Print "Hello World" from GPT/Claude. |

#### Day 2

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | How LLMs work (high level) | Read Ch 2. Understand tokens, context windows, temperature, top-p, and how models are trained (pre-training vs fine-tuning). You do NOT need to understand backpropagation in detail. |
| **Afternoon (Build)** | Play with the API | Build a simple script that takes user input, sends it to an LLM, and prints the response. Experiment with temperature and max tokens. Try different system prompts. |

**Short on Time?** Combine Days 1 and 2 into a single day. Skip the historical context in Ch 1 and jump straight to "What are LLMs?" and the API setup.

---

### Day 3: Transformers Intuition

**Reference:** [Chapter 3 -- Transformers](03-transformers.md)

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | The Transformer architecture | Read Ch 3. Focus on understanding **attention** ("what does the model pay attention to when generating the next word?"). Watch 3Blue1Brown's visual explanation if the text feels dense. You need intuition, not math. |
| **Afternoon (Build)** | Visualize attention | Use a tool like BertViz or the OpenAI tokenizer to see how models break text into tokens. Tokenize a few sentences and observe how different phrasings change the token count. This will help you understand context window limits later. |

**Short on Time?** Spend 1 hour reading the attention summary in Ch 3 and skip the build block. Transformers intuition is useful but you won't be coding a transformer from scratch.

---

### Day 4: Embeddings and Vector Databases

**Reference:** [Chapter 4 -- Embeddings & Vector DBs](04-embeddings-vectordb.md)

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | What are embeddings? | Read Ch 4. Understand that embeddings turn text into numbers (vectors) that capture meaning. Similar meanings = similar vectors. This is the foundation of search, RAG, and recommendations. |
| **Afternoon (Build)** | Build a semantic search tool | Use OpenAI's embedding API to embed 20-30 text snippets. Store them in a simple vector DB (start with ChromaDB -- it's the easiest). Build a script where you type a query and it returns the most relevant snippets. |

**Short on Time?** Do not skip this day. Embeddings and vector DBs are critical for everything that comes after. If you must cut something, keep the build block short but still do it.

---

### Days 5-6: Prompt Engineering + Beginner Projects

**Reference:** [Chapter 5 -- Prompt Engineering](05-prompt-engineering.md)

#### Day 5

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | Core prompt engineering techniques | Read Ch 5. Learn zero-shot, few-shot, chain-of-thought, and role-based prompting. Understand why prompt structure matters more than most people think. |
| **Afternoon (Build)** | Prompt engineering lab | Take one task (e.g., "summarize an article") and try 10 different prompt variations. Track which ones give the best output. Build a simple prompt template system. |

#### Day 6

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | Advanced prompting patterns | Finish Ch 5. Learn about output formatting (JSON mode), prompt chaining, and handling edge cases. Study examples of production-quality prompts. |
| **Afternoon (Build)** | Beginner project | Build one of these: (1) A blog post generator that takes a topic and tone, (2) A code explainer that takes code and explains it in plain English, or (3) A study buddy chatbot for a subject you're learning. |

**Short on Time?** Day 5 is essential. If you must compress, do Day 5 fully and merge Day 6's afternoon project into Day 7.

---

### Days 7-9: LangChain + Intermediate Projects

**Reference:** [Chapter 6 -- LangChain & Orchestration](06-langchain.md)

#### Day 7

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | LangChain fundamentals | Read Ch 6 (first half). Learn about chains, prompts, and output parsers. Understand why orchestration frameworks exist -- they save you from writing the same glue code over and over. |
| **Afternoon (Build)** | Your first chain | Build a LangChain pipeline that takes a topic, generates an outline, then expands each section into a paragraph. This is a simple sequential chain and it teaches you the core pattern. |

#### Day 8

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | Agents and tools | Read Ch 6 (second half). Learn how LangChain agents can use tools (web search, calculators, APIs) to answer questions they can't answer from training data alone. |
| **Afternoon (Build)** | Build an agent | Create a LangChain agent that can search the web and answer questions with citations. Use Tavily or SerpAPI for search. This is one of the most impressive demos you can build. |

#### Day 9

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | Review and fill gaps | Re-read any sections from Ch 4-6 that felt unclear. Look at LangChain's official cookbook examples for patterns you haven't tried. |
| **Afternoon (Build)** | Intermediate project | Build one of these: (1) A document Q&A bot (upload a PDF, ask questions about it), (2) A multi-step research agent that gathers info from multiple sources, or (3) A chatbot with memory that remembers previous conversations. |

**Short on Time?** Do Days 7 and 8. Skip Day 9's morning review and go straight to building the intermediate project.

---

### Days 10-12: RAG + Advanced Projects

**Reference:** [Chapter 7 -- RAG (Retrieval-Augmented Generation)](07-rag.md)

#### Day 10

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | RAG fundamentals | Read Ch 7 (first section). Understand the RAG pipeline: load documents, chunk them, embed them, store in a vector DB, retrieve relevant chunks, feed them to the LLM. This is the most important production pattern in GenAI. |
| **Afternoon (Build)** | Basic RAG pipeline | Build a RAG system from scratch (without LangChain first). Load a few text files, chunk them, embed with OpenAI, store in ChromaDB, and query. See how the LLM's answers improve when given relevant context. |

#### Day 11

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | Advanced RAG techniques | Read Ch 7 (second section). Learn about chunking strategies, hybrid search, re-ranking, metadata filtering, and how to evaluate RAG quality. |
| **Afternoon (Build)** | RAG with LangChain | Rebuild your RAG pipeline using LangChain's document loaders, text splitters, and retrievers. Compare the experience to your from-scratch version. Add a simple Streamlit or Gradio frontend. |

#### Day 12

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | Production RAG considerations | Finish Ch 7. Learn about handling large document sets, updating embeddings, dealing with stale data, and cost optimization. |
| **Afternoon (Build)** | Advanced project | Build one of these: (1) A "Chat with your codebase" tool, (2) A company knowledge base Q&A system, or (3) A research paper search and summarization tool. Use everything you've learned so far. |

**Short on Time?** Days 10-11 are non-negotiable. If you must cut Day 12, still build the advanced project -- just skip the morning reading and use Ch 7 as a reference while building.

---

### Day 13: End-to-End App Building

**Reference:** [Chapter 8 -- Building End-to-End Apps](08-end-to-end-apps.md)

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | From script to application | Read Ch 8. Learn about structuring a GenAI app: frontend (Streamlit/Gradio/Next.js), backend (FastAPI), LLM orchestration, vector DB, and deployment. Understand cost management and rate limiting. |
| **Afternoon (Build)** | Deploy something | Take your best project from the previous days and turn it into a deployable app. Add a proper UI, error handling, and deploy it (Streamlit Cloud, Railway, or Vercel are all free/cheap). Having a live URL is a huge resume booster. |

**Short on Time?** Focus on deployment. Even a simple Streamlit app deployed to Streamlit Cloud counts. The goal is to have something live that you can share.

---

### Day 14: Advanced Topics Overview

**Reference:** [Chapter 9 -- Advanced Topics](09-advanced-topics.md)

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | Survey the landscape | Read Ch 9. Get a high-level understanding of fine-tuning (LoRA/QLoRA), multi-modal models (vision + text), AI agents, function calling, and evaluation frameworks. You don't need to master these today -- just know they exist and when you'd use them. |
| **Afternoon (Build)** | Try one advanced feature | Pick one thing from Ch 9 and try it: (1) Use GPT-4 Vision or Claude's vision to analyze images, (2) Implement function calling to let the LLM trigger real actions, or (3) Set up a simple evaluation pipeline for your RAG system. |

**Short on Time?** Spend 1 hour skimming Ch 9 headings and summaries. Know the vocabulary (fine-tuning, LoRA, multi-modal, agents, evals) so you can speak intelligently about them in interviews, even if you haven't built with them yet.

---

### Day 15: Final Project + Interview Prep

**Reference:** [Chapter 10 -- Projects & Interview Prep](10-projects-interview.md)

| Block | Focus | Details |
|-------|-------|---------|
| **Morning (Learn)** | Interview prep | Read Ch 10. Review common GenAI interview questions. Practice explaining RAG, embeddings, prompt engineering, and transformer attention in simple terms. If you can explain it to a non-technical friend, you understand it well enough. |
| **Afternoon (Build)** | Polish your portfolio | Pick your 2-3 best projects. Clean up the code, write clear READMEs, and push to GitHub. Update your resume/LinkedIn with your GenAI projects. Record a 2-minute demo video if you can. |

**Short on Time?** Do not skip interview prep. Even 1 hour of reviewing key concepts and practicing explanations will make a big difference. Focus on the "explain it simply" exercise.

---

## Daily Schedule Template

Here's a concrete schedule you can copy into your calendar:

```
09:00 - 09:15  Review yesterday's notes, set today's goal
09:15 - 11:30  LEARN: Read the chapter, watch supplementary videos
11:30 - 12:00  Summarize key takeaways in your own words (3-5 bullet points)
12:00 - 13:00  Lunch break (seriously, take the break)
13:00 - 15:30  BUILD: Code the day's project
15:30 - 16:00  Push code to GitHub, write a short "what I built today" note
```

---

## Tips for Staying on Track

1. **Build every single day.** Reading without building is the fastest way to forget everything. Even 30 minutes of coding counts.

2. **Don't get stuck in tutorial hell.** If a concept isn't clicking after 30 minutes of reading, move on and build something. Understanding often comes from doing, not reading.

3. **Use AI to learn AI.** Stuck on a concept? Ask ChatGPT or Claude to explain it. Debugging an error? Paste it into an LLM. You're learning these tools -- use them.

4. **Keep a learning journal.** Spend 5 minutes at the end of each day writing what you learned and what confused you. This is incredibly useful for interview prep later.

5. **Ship something ugly.** Your first project will not be beautiful. That's fine. A working ugly app beats a theoretical perfect one every time.

6. **Find one accountability partner.** Share your daily progress with a friend, study group, or even on Twitter/LinkedIn. Public commitment works.

7. **Don't compare your Day 3 to someone else's Day 300.** Everyone starts somewhere. Focus on your own progress.

8. **When in doubt, build a RAG app.** It's the most in-demand skill, it touches every concept (embeddings, vector DBs, prompting, LLMs), and it's impressive in interviews.

---

## Quick Reference: Chapter Map

| Day(s) | Chapter | Core Skill |
|--------|---------|------------|
| 1-2 | [Ch 1: Foundations](01-foundations.md), [Ch 2: LLM Basics](02-llm-basics.md) | Understanding GenAI + first API calls |
| 3 | [Ch 3: Transformers](03-transformers.md) | Attention intuition |
| 4 | [Ch 4: Embeddings & Vector DBs](04-embeddings-vectordb.md) | Semantic search |
| 5-6 | [Ch 5: Prompt Engineering](05-prompt-engineering.md) | Prompting techniques + first project |
| 7-9 | [Ch 6: LangChain & Orchestration](06-langchain.md) | Chains, agents, tools |
| 10-12 | [Ch 7: RAG](07-rag.md) | Production RAG pipelines |
| 13 | [Ch 8: End-to-End Apps](08-end-to-end-apps.md) | Deployment |
| 14 | [Ch 9: Advanced Topics](09-advanced-topics.md) | Fine-tuning, multi-modal, evals |
| 15 | [Ch 10: Projects & Interview Prep](10-projects-interview.md) | Portfolio + interview readiness |

---

## What "Done" Looks Like

By Day 15, you should have:

- A working understanding of how LLMs, embeddings, and RAG work
- Hands-on experience with OpenAI/Anthropic APIs and LangChain
- At least 3 projects on GitHub (beginner, intermediate, advanced)
- At least 1 deployed application with a live URL
- Confidence to discuss GenAI concepts in interviews
- A foundation to keep learning advanced topics on your own

You won't be an expert. But you'll be dangerous enough to build real things, contribute to GenAI teams, and keep learning on the job. That's the goal.

**Now stop reading this plan and go start Day 1.**
