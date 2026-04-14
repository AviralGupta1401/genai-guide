# Chapter 7: Retrieval-Augmented Generation (RAG)

> "An LLM without RAG is like a brilliant student taking a closed-book exam on material they never studied. RAG hands them the textbook."

---

## 1. Concept Explanation

### What Is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances Large Language Models by giving them access to external knowledge at query time. Instead of relying solely on what the model learned during training, RAG first *retrieves* relevant documents from a knowledge base, then *feeds* that information into the LLM's prompt so it can generate a grounded, accurate answer.

### Why Do LLMs Need RAG?

LLMs are powerful, but they have three fundamental limitations:

1. **Knowledge Cutoff** -- Every LLM has a training cutoff date. GPT-4 doesn't know about events after its training. Claude doesn't know your company's Q4 2025 earnings. The world moves on; the model stays frozen.

2. **Hallucination** -- When an LLM doesn't know something, it often makes up a plausible-sounding but incorrect answer rather than admitting ignorance. This is dangerous in production systems.

3. **No Access to Private Data** -- Your company's internal wiki, HR policies, proprietary research, personal notes -- none of this was in the training data. The LLM literally cannot answer questions about it.

RAG solves all three problems by retrieving up-to-date, relevant, private information and injecting it into the prompt.

### The Open-Book Exam Analogy

Think about exams in college:

- **Closed-book exam** = a vanilla LLM. It can only use what it memorized (its training data). If it didn't study a topic, it guesses.
- **Open-book exam** = RAG. The student (LLM) gets to look up relevant pages in the textbook (knowledge base) before answering. The answer is grounded in real source material.

The student still needs to be smart enough to understand the material and synthesize a good answer -- but now they have facts to work with instead of just memory.

### The RAG Pipeline at a Glance

```
Ingest --> Retrieve --> Generate
```

1. **Ingest**: Load your documents, break them into chunks, convert chunks to embeddings, store in a vector database.
2. **Retrieve**: When a user asks a question, convert the question to an embedding, search the vector database for similar chunks.
3. **Generate**: Pass the retrieved chunks as context to the LLM, which generates an answer grounded in that context.

---

## 2. Detailed Notes (Book Style)

### The RAG Pipeline in Detail

#### Phase 1: Indexing (Offline / One-Time)

**Step 1: Document Loading**

Raw data comes in many formats -- PDFs, Word documents, web pages, CSVs, databases, APIs. Document loaders parse these into plain text that the pipeline can process.

```
PDF / DOCX / HTML / CSV  -->  Document Loader  -->  Raw Text
```

**Step 2: Chunking (Text Splitting)**

A full document is too large to fit in an LLM's context window and too broad to be useful for retrieval. We split it into smaller, semantically meaningful pieces called *chunks*.

**Chunking Strategies:**

| Strategy | How It Works | Best For |
|---|---|---|
| **Fixed-size** | Split every N characters (e.g., 500 chars) with overlap | Simple, predictable |
| **Recursive** | Try splitting by paragraphs, then sentences, then words | General-purpose (recommended default) |
| **Semantic** | Use embeddings to detect topic boundaries | High-quality retrieval when structure varies |
| **Document-based** | Split by natural boundaries (pages, sections, headers) | Structured documents like manuals |

**Key parameters:**
- `chunk_size`: Number of characters per chunk (typically 500-1500)
- `chunk_overlap`: Characters shared between consecutive chunks (typically 50-200). Overlap prevents cutting a sentence in half and losing context.

**Step 3: Embedding**

Each chunk is converted into a high-dimensional vector (a list of numbers, typically 768 or 1536 dimensions) using an embedding model. These vectors capture the *semantic meaning* of the text -- chunks about similar topics end up close together in vector space.

Popular embedding models:
- `text-embedding-3-small` (OpenAI) -- good quality, affordable
- `text-embedding-3-large` (OpenAI) -- higher quality
- `all-MiniLM-L6-v2` (open source, via Sentence Transformers) -- free, runs locally

**Step 4: Storing in a Vector Database**

The embeddings are stored in a specialized database optimized for similarity search. Unlike traditional databases that search by exact match, vector databases find the *most similar* vectors to a query vector.

Popular vector databases:
- **ChromaDB** -- simple, local, great for learning and prototyping
- **Pinecone** -- managed cloud service, scales well
- **Weaviate** -- open source, feature-rich
- **FAISS** -- Facebook's library, fast but lower-level

#### Phase 2: Query (Online / Per-Request)

**Step 5: Query Embedding**

The user's question is converted to an embedding using the *same* embedding model used during indexing.

**Step 6: Retrieval**

The query embedding is compared against all stored chunk embeddings to find the most relevant chunks.

**Retrieval Methods:**

| Method | Description | Pros | Cons |
|---|---|---|---|
| **Similarity Search** | Find the K nearest vectors (cosine similarity) | Simple, fast | May return redundant results |
| **MMR (Maximal Marginal Relevance)** | Balance relevance with diversity | Reduces redundancy | Slightly slower |
| **Hybrid Search** | Combine vector search with keyword search (BM25) | Best of both worlds | More complex setup |

**Step 7: Context Injection (Prompt Construction)**

The retrieved chunks are formatted and inserted into the LLM's prompt:

```
System: You are a helpful assistant. Answer the question based ONLY
on the following context. If the context doesn't contain the answer,
say "I don't have enough information to answer that."

Context:
{chunk_1}
{chunk_2}
{chunk_3}

User: {original_question}
```

**Step 8: LLM Generation**

The LLM reads the context and generates an answer grounded in the retrieved information.

### RAG Evaluation Metrics

How do you know if your RAG pipeline is working well?

| Metric | What It Measures | How to Compute |
|---|---|---|
| **Faithfulness** | Is the answer supported by the retrieved context? | Check if claims in the answer appear in the context |
| **Answer Relevance** | Does the answer address the question? | Semantic similarity between question and answer |
| **Context Relevance** | Are the retrieved chunks actually relevant? | Semantic similarity between question and each chunk |
| **Context Recall** | Did retrieval find all the necessary information? | Compare retrieved chunks against a ground-truth answer |

Tools like **RAGAS** (Retrieval Augmented Generation Assessment) automate these evaluations.

### RAG vs. Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|---|---|---|
| **Use case** | Access external/changing data | Learn a specific style or task |
| **Data freshness** | Always up-to-date | Frozen at training time |
| **Cost** | Vector DB + retrieval overhead | GPU training cost |
| **Transparency** | Can cite sources | No source tracking |
| **Setup complexity** | Moderate | High |
| **Best for** | Q&A over documents, customer support, search | Code generation in a specific framework, tone adaptation |

**Rule of thumb**: If you need the model to *know* new facts, use RAG. If you need the model to *behave* differently, use fine-tuning. You can also combine both.

### Advanced RAG Patterns

**Multi-Query Retrieval**: The LLM generates multiple reformulations of the user's question, retrieves documents for each, and merges the results. This catches relevant chunks that a single query phrasing might miss.

**Contextual Compression**: After retrieval, a secondary LLM pass extracts only the relevant sentences from each chunk, reducing noise in the context.

**Parent-Child Retrieval**: Index small chunks (for precise retrieval) but return the larger parent chunk (for full context). You get the best of both worlds -- precise matching and sufficient context.

**Self-Querying**: The LLM parses the user's question to extract structured filters (e.g., "papers from 2024 about transformers" becomes a metadata filter for `year=2024` AND `topic=transformers` combined with a vector search for the content).

---

## 3. Visual / Intuitive Explanation

Let us walk through a concrete example. Imagine you are building a chatbot that answers questions about your company's HR policies, which live in a 50-page PDF.

### Indexing Phase (One-Time Setup)

```
                         INDEXING PHASE
 +-------------+     +------------+     +-------------+
 |  HR Policy  | --> | Document   | --> | Text        |
 |  PDF (50pg) |     | Loader     |     | Splitter    |
 +-------------+     +------------+     +------+------+
                                               |
                              Chunks: ["Employees get 20 days PTO...",
                                        "Parental leave policy...",
                                        "Remote work guidelines...",
                                        ... 150 chunks total]
                                               |
                                               v
                                      +-----------------+
                                      | Embedding Model |
                                      +-----------------+
                                               |
                              Vectors: [[0.12, -0.45, 0.78, ...],
                                         [0.33, 0.21, -0.56, ...],
                                         ...]
                                               |
                                               v
                                      +-----------------+
                                      |   ChromaDB      |
                                      | (Vector Store)  |
                                      +-----------------+
```

**What happened**: The PDF was loaded, split into ~150 chunks of ~500 characters each, each chunk was converted to a vector, and all vectors were stored in ChromaDB.

### Query Phase (Every Time a User Asks)

```
                          QUERY PHASE

  User: "How many vacation days do I get?"
                     |
                     v
            +-----------------+
            | Embedding Model |  (same model as indexing!)
            +-----------------+
                     |
            Query vector: [0.14, -0.42, 0.81, ...]
                     |
                     v
            +-----------------+
            |   ChromaDB      |  -- cosine similarity search -->
            | (Vector Store)  |
            +-----------------+
                     |
            Top 3 matching chunks:
            1. "Employees get 20 days of paid time off (PTO)
                per year. PTO accrues at 1.67 days per month..."
            2. "Unused PTO can be carried over up to 5 days
                into the next calendar year..."
            3. "Part-time employees receive prorated PTO
                based on hours worked..."
                     |
                     v
            +-----------------+
            | Prompt Template |
            +-----------------+
            "Answer based on this context:
             {chunk_1} {chunk_2} {chunk_3}
             Question: How many vacation days do I get?"
                     |
                     v
            +-----------------+
            |   LLM (Claude)  |
            +-----------------+
                     |
                     v
            "Based on the HR policy, full-time employees
             receive 20 days of paid time off (PTO) per year,
             which accrues at 1.67 days per month. Up to 5
             unused days can be carried over to the next year."
```

The key insight: the LLM never read the full 50-page PDF. It only saw the 3 most relevant chunks. This is efficient, focused, and keeps the answer grounded in actual policy text.

---

## 4. YouTube Resources

Search for these on YouTube to find up-to-date video tutorials:

| Search Term | What You Will Learn |
|---|---|
| `"RAG explained in 10 minutes"` | High-level intuition behind RAG |
| `"Build a RAG chatbot with LangChain"` | Hands-on tutorial building a complete pipeline |
| `"LangChain RAG tutorial 2025"` | Latest LangChain patterns for RAG |
| `"ChromaDB tutorial Python"` | How to use ChromaDB as your vector store |
| `"RAG vs fine-tuning when to use"` | Decision framework for choosing the right approach |
| `"Advanced RAG techniques"` | Multi-query, re-ranking, hybrid search |

**Recommended channels**: search for content from channels that focus on applied AI/ML engineering and LLM application development.

---

## 5. Official Documentation

### Must-Read Documentation

| Resource | URL | What to Focus On |
|---|---|---|
| **LangChain RAG Tutorial** | docs.langchain.com | "Build a RAG App" tutorial under Use Cases |
| **LangChain Text Splitters** | docs.langchain.com | Different chunking strategies and parameters |
| **ChromaDB Docs** | docs.trychroma.com | Getting Started, Collections, Querying |
| **LlamaIndex Docs** | docs.llamaindex.ai | Alternative to LangChain; great "Starter Tutorial" |
| **OpenAI Embeddings Guide** | platform.openai.com | Understanding embedding models and usage |
| **RAGAS Docs** | docs.ragas.io | RAG evaluation framework |

### Reading Order for Beginners

1. Start with the LangChain "Build a RAG App" quickstart
2. Read ChromaDB's Getting Started guide
3. Work through the code examples in this chapter
4. Explore LlamaIndex as an alternative framework
5. Learn evaluation with RAGAS once your pipeline works

---

## 6. Code Examples

### Prerequisites

```bash
pip install langchain langchain-openai langchain-anthropic langchain-chroma
pip install chromadb pypdf tiktoken
```

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
```

### Step 1: Load Documents

```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load a PDF
pdf_loader = PyPDFLoader("company_handbook.pdf")
pdf_docs = pdf_loader.load()

print(f"Loaded {len(pdf_docs)} pages from PDF")
print(f"First page preview: {pdf_docs[0].page_content[:200]}")

# Load a text file
text_loader = TextLoader("notes.txt")
text_docs = text_loader.load()

# Each document has: page_content (str) and metadata (dict)
print(pdf_docs[0].metadata)
# {'source': 'company_handbook.pdf', 'page': 0}
```

### Step 2: Split Text into Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Target size of each chunk in characters
    chunk_overlap=100,     # Overlap between chunks to preserve context
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try splitting by these, in order
)

chunks = splitter.split_documents(pdf_docs)

print(f"Split {len(pdf_docs)} pages into {len(chunks)} chunks")
print(f"\nExample chunk:\n{chunks[0].page_content}")
print(f"\nChunk metadata: {chunks[0].metadata}")
```

### Step 3: Create Embeddings and Store in ChromaDB

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Initialize the embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create vector store from chunks (this embeds and stores in one step)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",   # Save to disk
    collection_name="company_docs"
)

print(f"Stored {vectorstore._collection.count()} chunks in ChromaDB")
```

### Step 4: Query with Similarity Search

```python
# Simple similarity search
query = "What is the remote work policy?"
results = vectorstore.similarity_search(query, k=3)

print(f"Found {len(results)} relevant chunks:\n")
for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} (from {doc.metadata.get('source', 'unknown')}) ---")
    print(doc.page_content)
    print()

# Similarity search with scores
results_with_scores = vectorstore.similarity_search_with_score(query, k=3)
for doc, score in results_with_scores:
    print(f"Score: {score:.4f} | {doc.page_content[:80]}...")
```

### Step 5: Build the Full RAG Chain (OpenAI)

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# Create the retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Define the prompt template
template = """Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say "I don't have
enough information to answer that question."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Helper to format retrieved docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask a question
answer = rag_chain.invoke("What is the remote work policy?")
print(answer)
```

### Step 6: RAG Chain with Claude

```python
from langchain_anthropic import ChatAnthropic

# Swap the LLM -- everything else stays the same
llm_claude = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

rag_chain_claude = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_claude
    | StrOutputParser()
)

answer = rag_chain_claude.invoke("How many sick days do employees get?")
print(answer)
```

### Step 7: Evaluate Retrieval Quality

```python
def evaluate_retrieval(query, expected_keywords, vectorstore, k=3):
    """Simple retrieval quality check."""
    results = vectorstore.similarity_search(query, k=k)
    retrieved_text = " ".join([doc.page_content for doc in results])

    # Check how many expected keywords appear in retrieved chunks
    found = []
    missing = []
    for keyword in expected_keywords:
        if keyword.lower() in retrieved_text.lower():
            found.append(keyword)
        else:
            missing.append(keyword)

    recall = len(found) / len(expected_keywords) if expected_keywords else 0

    print(f"Query: {query}")
    print(f"Recall: {recall:.0%} ({len(found)}/{len(expected_keywords)})")
    print(f"Found: {found}")
    print(f"Missing: {missing}")
    return recall

# Test it
evaluate_retrieval(
    query="What is the vacation policy?",
    expected_keywords=["PTO", "20 days", "carry over", "accrual"],
    vectorstore=vectorstore
)
```

### Using MMR Retrieval (Reducing Redundancy)

```python
# MMR balances relevance with diversity in results
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,        # Fetch 10 candidates, pick 4 diverse ones
        "lambda_mult": 0.7     # 0 = max diversity, 1 = max relevance
    }
)

results = retriever_mmr.invoke("employee benefits")
for doc in results:
    print(doc.page_content[:100], "...\n")
```

---

## 7. Mini Practice Tasks

### Task 1: Chunk Size Experiment
Load any text file and split it with chunk sizes of 200, 500, and 1000 characters. Print the number of chunks for each. Ask the same question with each configuration. Which gives the best answer?

### Task 2: Build a FAQ Retriever
Create a list of 10 question-answer pairs on any topic. Store them in ChromaDB. Build a retriever that, given a new question, finds the most similar existing question and returns its answer.

### Task 3: Compare Retrieval Methods
Using the same vector store, compare results from `similarity`, `mmr`, and `similarity_search_with_score` for 3 different queries. Document which method returns more diverse and relevant results.

### Task 4: Source Citation
Modify the RAG chain prompt to instruct the LLM to cite which chunk(s) it used in its answer. Include page numbers and source filenames from the metadata.

### Task 5: Multi-Document RAG
Load 3 different documents (e.g., a PDF, a text file, and a CSV). Store all of them in the same vector store with proper metadata. Ask questions that require information from specific documents and verify the retriever finds the right source.

---

## 8. Quick Revision Summary

| Concept | Key Point |
|---|---|
| **RAG** | Retrieve relevant docs, inject as context, then generate |
| **Why RAG** | Solves knowledge cutoff, hallucination, and private data access |
| **Chunking** | Split documents into small pieces (500-1000 chars typical) |
| **Embeddings** | Convert text to vectors that capture semantic meaning |
| **Vector DB** | Database optimized for finding similar vectors (ChromaDB, Pinecone) |
| **Retrieval** | Similarity search, MMR (for diversity), or hybrid (vector + keyword) |
| **Context injection** | Put retrieved chunks into the LLM prompt |
| **Evaluation** | Faithfulness, relevance, recall -- use RAGAS for automation |
| **RAG vs Fine-tuning** | RAG for new knowledge, fine-tuning for new behavior |
| **Advanced patterns** | Multi-query, compression, parent-child, self-querying |
| **Chunk overlap** | Prevents losing context at chunk boundaries |

**The one-liner**: RAG = Embed your docs, find relevant chunks for the query, paste them into the prompt, let the LLM answer.

---

## 9. Common Mistakes

### Mistake 1: Wrong Chunk Size
**Problem**: Chunks too small lose context; chunks too large dilute relevance and waste tokens.
**Fix**: Start with 500-1000 characters and 100-200 overlap. Experiment and evaluate. There is no universal best size -- it depends on your documents.

### Mistake 2: Ignoring Retrieval Quality
**Problem**: Developers spend all their time tuning the LLM prompt but never check if the retriever is returning relevant chunks. Garbage in, garbage out.
**Fix**: Always inspect retrieved chunks before passing them to the LLM. Build evaluation scripts. If retrieval is bad, no amount of prompt engineering will fix it.

### Mistake 3: Using Different Embedding Models for Indexing and Querying
**Problem**: You index with `text-embedding-3-small` but query with `text-embedding-3-large`. The vector spaces are incompatible, so similarity search returns random results.
**Fix**: Always use the exact same embedding model for both indexing and querying.

### Mistake 4: Not Enough Context in the Prompt
**Problem**: Retrieving only 1 chunk when the answer spans multiple sections of the document.
**Fix**: Retrieve 3-5 chunks (adjust `k`). Use chunk overlap so related information is captured. Consider parent-child retrieval for broader context.

### Mistake 5: No Fallback for Unanswerable Questions
**Problem**: The LLM hallucinates an answer when the retrieved context doesn't actually contain the information.
**Fix**: Explicitly instruct the LLM in the prompt: "If the context does not contain the answer, say you don't know." Test with questions you know are NOT in the knowledge base.

### Mistake 6: Forgetting Metadata
**Problem**: You store chunks but lose track of which document, page, or section they came from. Users ask "where did you get that?" and you cannot answer.
**Fix**: Always preserve and pass metadata (source file, page number, section heading) through the pipeline. Include it in your prompt template for citations.

### Mistake 7: Never Updating the Knowledge Base
**Problem**: Your documents change, but the vector store still has the old versions. Users get outdated answers.
**Fix**: Build a re-indexing pipeline. When documents change, delete old chunks and re-ingest. Consider timestamping chunks and filtering by recency.

### Mistake 8: Loading Entire Documents Without Preprocessing
**Problem**: PDFs with headers, footers, page numbers, and boilerplate pollute your chunks with noise.
**Fix**: Clean your documents before chunking. Strip repeated headers/footers, remove page numbers, and handle tables and images appropriately.

---

## Advanced Projects

---

### Project 1: PDF Chatbot

#### Problem Statement
Build a web application where users can upload any PDF file and have a conversation with it. The app should use RAG to retrieve relevant sections from the PDF and generate accurate answers with source citations.

#### Tech Stack
- **Frontend**: Streamlit
- **RAG Framework**: LangChain
- **Vector Store**: ChromaDB (in-memory)
- **LLM**: OpenAI GPT-4o-mini (easily swappable to Claude)
- **Embeddings**: OpenAI text-embedding-3-small

#### Architecture

```
User uploads PDF
       |
       v
 PyPDFLoader --> RecursiveCharacterTextSplitter --> OpenAIEmbeddings
       |                                                   |
       v                                                   v
  Raw pages                                         ChromaDB (in-memory)
                                                           |
User asks question  ------>  Retriever (top 4 chunks) -----+
       |                           |
       v                           v
  Prompt Template  <--  Retrieved chunks + question
       |
       v
   LLM generates answer with citations
       |
       v
  Streamlit displays answer + sources
```

#### Full Code

```python
# pdf_chatbot.py
import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- Configuration ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVAL_K = 4
LLM_MODEL = "gpt-4o-mini"

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Chatbot", page_icon="📄")
st.title("PDF Chatbot")
st.caption("Upload a PDF and ask questions about its contents.")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: API key and file upload
with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file and api_key:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                os.environ["OPENAI_API_KEY"] = api_key

                # Save uploaded file to a temp path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                # Load and split
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP
                )
                chunks = splitter.split_documents(pages)

                # Create vector store
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings
                )
                st.session_state.chat_history = []

                # Clean up temp file
                os.unlink(tmp_path)

            st.success(f"Processed {len(pages)} pages into {len(chunks)} chunks.")

# Chat interface
if st.session_state.vectorstore:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask a question about the PDF..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Build RAG chain
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": RETRIEVAL_K}
        )

        template = """You are a helpful assistant that answers questions based on
the provided document context. Always cite the page number when possible.

Context:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the context above.
- If the answer is not in the context, say "I could not find this information in the document."
- Cite page numbers in parentheses, e.g., (Page 3).

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

        def format_docs(docs):
            formatted = []
            for doc in docs:
                page_num = doc.metadata.get("page", "?")
                formatted.append(f"[Page {page_num}]: {doc.page_content}")
            return "\n\n".join(formatted)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = rag_chain.invoke(user_input)
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
else:
    st.info("Upload a PDF and enter your API key to get started.")
```

#### How to Run

```bash
pip install streamlit langchain langchain-openai langchain-chroma chromadb pypdf
streamlit run pdf_chatbot.py
```

#### Extensions
- Add support for multiple PDFs simultaneously
- Add conversation memory so follow-up questions work
- Switch to Claude by replacing `ChatOpenAI` with `ChatAnthropic`
- Add a "Show Sources" expander to display the raw retrieved chunks
- Export chat history as a text file

---

### Project 2: Personalized Learning Assistant

#### Problem Statement
Build an AI tutor that takes a topic, retrieves relevant information from a knowledge base of educational content, generates personalized explanations at the user's level, and creates practice quizzes to test understanding.

#### Tech Stack
- **Framework**: LangChain
- **Vector Store**: ChromaDB (persistent)
- **LLM**: Anthropic Claude
- **Embeddings**: OpenAI text-embedding-3-small
- **UI**: Command-line (terminal)

#### Architecture

```
Knowledge Base (text files on various topics)
       |
       v
  Indexing Pipeline --> ChromaDB (persistent)
       |
       v
  User selects topic + difficulty level
       |
       v
  Retriever fetches relevant content
       |
       v
  LLM generates: 1) Explanation  2) Quiz  3) Feedback on answers
```

#### Full Code

```python
# learning_assistant.py
import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- Configuration ---
KNOWLEDGE_DIR = "./knowledge_base"     # Directory of .txt files
CHROMA_DIR = "./learning_chroma_db"
COLLECTION_NAME = "learning_materials"

os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"


def build_knowledge_base():
    """Load documents from knowledge_base/ and index them."""
    print("Building knowledge base...")

    # Create sample content if directory is empty
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    sample_files = {
        "python_basics.txt": (
            "Python Basics\n\n"
            "Python is a high-level, interpreted programming language. "
            "Variables in Python don't need type declarations. "
            "Python uses indentation to define code blocks instead of braces. "
            "Lists are ordered, mutable collections created with square brackets. "
            "Dictionaries store key-value pairs created with curly braces. "
            "Functions are defined with the 'def' keyword. "
            "Python supports list comprehensions for concise list creation. "
            "The 'for' loop iterates over sequences. "
            "Python has built-in support for object-oriented programming with classes."
        ),
        "machine_learning.txt": (
            "Machine Learning Fundamentals\n\n"
            "Machine learning is a subset of AI where systems learn from data. "
            "Supervised learning uses labeled data to train models. "
            "Unsupervised learning finds patterns in unlabeled data. "
            "Common algorithms include linear regression, decision trees, and neural networks. "
            "Overfitting occurs when a model memorizes training data instead of learning patterns. "
            "Cross-validation helps evaluate model performance on unseen data. "
            "Feature engineering is the process of selecting and transforming input variables. "
            "Gradient descent is an optimization algorithm used to minimize loss functions."
        ),
        "data_structures.txt": (
            "Data Structures\n\n"
            "Arrays store elements in contiguous memory locations with O(1) access. "
            "Linked lists consist of nodes where each node points to the next. "
            "Stacks follow Last-In-First-Out (LIFO) principle. "
            "Queues follow First-In-First-Out (FIFO) principle. "
            "Binary trees have at most two children per node. "
            "Hash tables provide O(1) average-case lookup using hash functions. "
            "Graphs represent relationships between entities with vertices and edges. "
            "A heap is a complete binary tree where parent nodes satisfy the heap property."
        ),
    }

    for filename, content in sample_files.items():
        filepath = os.path.join(KNOWLEDGE_DIR, filename)
        if not os.path.exists(filepath):
            with open(filepath, "w") as f:
                f.write(content)

    # Load all text files
    loader = DirectoryLoader(KNOWLEDGE_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME
    )
    print(f"Knowledge base ready with {vectorstore._collection.count()} chunks.\n")
    return vectorstore


def load_knowledge_base():
    """Load existing vector store from disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )


def generate_explanation(vectorstore, topic, level):
    """Retrieve relevant content and generate a personalized explanation."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(topic)

    context = "\n\n".join(doc.page_content for doc in docs)

    template = """You are a patient, encouraging tutor. Explain the topic
to a student at the {level} level.

Reference material:
{context}

Topic to explain: {topic}

Instructions:
- Use the reference material as your knowledge source.
- Adjust complexity to the {level} level.
- Use analogies and examples.
- For "beginner": assume no prior knowledge, use simple language.
- For "intermediate": assume basic understanding, go deeper.
- For "advanced": assume solid foundations, discuss nuances and edge cases.
- Keep the explanation focused and structured.

Explanation:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.3)

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "topic": topic, "level": level})


def generate_quiz(vectorstore, topic, num_questions=3):
    """Generate a quiz based on retrieved content."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(topic)
    context = "\n\n".join(doc.page_content for doc in docs)

    template = """Based on the following material, create a quiz with
{num_questions} multiple-choice questions.

Material:
{context}

Topic: {topic}

Format each question as:
Q1: [Question text]
A) [Option]
B) [Option]
C) [Option]
D) [Option]
Correct: [Letter]

Make questions that test understanding, not just memorization.

Quiz:"""

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.5)

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "context": context,
        "topic": topic,
        "num_questions": num_questions
    })


def main():
    """Main interactive loop."""
    # Build or load knowledge base
    if os.path.exists(CHROMA_DIR):
        print("Loading existing knowledge base...")
        vectorstore = load_knowledge_base()
    else:
        vectorstore = build_knowledge_base()

    print("=== Personalized Learning Assistant ===\n")
    print("Available commands:")
    print("  learn <topic>    - Get a personalized explanation")
    print("  quiz <topic>     - Take a quiz on a topic")
    print("  quit             - Exit\n")

    level = input("What is your level? (beginner/intermediate/advanced): ").strip()
    if level not in ("beginner", "intermediate", "advanced"):
        level = "beginner"
    print(f"Level set to: {level}\n")

    while True:
        user_input = input("\n> ").strip()
        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Happy learning!")
            break

        parts = user_input.split(" ", 1)
        command = parts[0].lower()
        topic = parts[1] if len(parts) > 1 else ""

        if command == "learn" and topic:
            print(f"\nGenerating explanation for '{topic}' at {level} level...\n")
            explanation = generate_explanation(vectorstore, topic, level)
            print(explanation)

        elif command == "quiz" and topic:
            print(f"\nGenerating quiz on '{topic}'...\n")
            quiz = generate_quiz(vectorstore, topic)
            print(quiz)

        else:
            print("Usage: 'learn <topic>' or 'quiz <topic>' or 'quit'")


if __name__ == "__main__":
    main()
```

#### How to Run

```bash
pip install langchain langchain-openai langchain-anthropic langchain-chroma chromadb
python learning_assistant.py
```

Example session:
```
What is your level? (beginner/intermediate/advanced): beginner

> learn python variables
> quiz data structures
> quit
```

#### Extensions
- Add a Streamlit web interface
- Track which topics the user has studied and quiz scores
- Add adaptive difficulty (increase level if quiz scores are high)
- Let users upload their own study materials to expand the knowledge base
- Add spaced repetition reminders for previously studied topics

---

### Project 3: Knowledge Base System with Source Citations

#### Problem Statement
Build a multi-document knowledge base system that ingests PDFs, text files, and CSVs, stores them in a persistent ChromaDB instance, and answers questions with proper source citations. Support adding new documents, listing sources, and querying across all documents.

#### Tech Stack
- **Framework**: LangChain
- **Vector Store**: ChromaDB (persistent on disk)
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **File Support**: PDF, TXT, CSV

#### Architecture

```
 Documents (PDF, TXT, CSV)
        |
        v
 +------------------+
 | Document Loaders |  (PyPDFLoader, TextLoader, CSVLoader)
 +--------+---------+
          |
          v
 +------------------+
 | Text Splitter    |  Metadata preserved: source, page, file type
 +--------+---------+
          |
          v
 +------------------+
 | OpenAI Embeddings|
 +--------+---------+
          |
          v
 +------------------+
 |   ChromaDB       |  Persistent storage at ./kb_chroma_db
 | (with metadata)  |
 +--------+---------+
          |
   Query with filters (optional: filter by source/type)
          |
          v
 +------------------+
 | LLM + Citations  |  Answer includes [Source: filename, Page: N]
 +------------------+
```

#### Full Code

```python
# knowledge_base.py
import os
import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# --- Configuration ---
CHROMA_DIR = "./kb_chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

os.environ["OPENAI_API_KEY"] = "your-openai-key"


class KnowledgeBase:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vectorstore = self._load_or_create_store()

    def _load_or_create_store(self):
        """Load existing ChromaDB or create a new one."""
        return Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )

    def _get_loader(self, file_path):
        """Return the appropriate loader based on file extension."""
        ext = Path(file_path).suffix.lower()
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".csv": CSVLoader,
        }
        loader_class = loaders.get(ext)
        if not loader_class:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders.keys())}")
        return loader_class(file_path)

    def add_document(self, file_path):
        """Ingest a document into the knowledge base."""
        file_path = str(Path(file_path).resolve())
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return

        print(f"Loading: {file_path}")
        loader = self._get_loader(file_path)
        documents = loader.load()

        # Add file type metadata
        file_type = Path(file_path).suffix.lower()
        file_name = Path(file_path).name
        for doc in documents:
            doc.metadata["file_type"] = file_type
            doc.metadata["file_name"] = file_name

        # Split into chunks
        chunks = self.splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")

        # Add to vector store
        self.vectorstore.add_documents(chunks)
        print(f"Added to knowledge base. Total chunks: {self.vectorstore._collection.count()}")

    def list_sources(self):
        """List all unique sources in the knowledge base."""
        results = self.vectorstore._collection.get(include=["metadatas"])
        if not results["metadatas"]:
            print("Knowledge base is empty.")
            return

        sources = set()
        for metadata in results["metadatas"]:
            name = metadata.get("file_name", metadata.get("source", "unknown"))
            sources.add(name)

        print(f"\nDocuments in knowledge base ({len(sources)} files):")
        for source in sorted(sources):
            print(f"  - {source}")
        print(f"\nTotal chunks: {self.vectorstore._collection.count()}")

    def query(self, question, k=4):
        """Query the knowledge base and return an answer with citations."""
        if self.vectorstore._collection.count() == 0:
            print("Knowledge base is empty. Add documents first.")
            return

        # Retrieve relevant chunks
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": k * 3}
        )

        def format_docs_with_sources(docs):
            formatted = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("file_name", doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "N/A")
                formatted.append(
                    f"[Source {i+1}: {source}, Page: {page}]\n{doc.page_content}"
                )
            return "\n\n---\n\n".join(formatted)

        template = """You are a knowledgeable assistant that answers questions
using ONLY the provided source documents. You MUST cite your sources.

Source Documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided sources.
- After each claim, cite the source in brackets, e.g., [Source 1: report.pdf, Page: 3].
- If the sources don't contain enough information, explicitly state what is missing.
- Be concise but thorough.

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {
                "context": retriever | format_docs_with_sources,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        print(f"\nSearching knowledge base for: '{question}'\n")
        answer = rag_chain.invoke(question)
        print(answer)

        # Also show retrieved chunks for transparency
        print("\n--- Retrieved Sources ---")
        docs = retriever.invoke(question)
        for i, doc in enumerate(docs):
            source = doc.metadata.get("file_name", doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "N/A")
            preview = doc.page_content[:120].replace("\n", " ")
            print(f"  [{i+1}] {source} (Page {page}): {preview}...")

    def clear(self):
        """Delete all documents from the knowledge base."""
        import shutil
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        self.vectorstore = self._load_or_create_store()
        print("Knowledge base cleared.")


def main():
    kb = KnowledgeBase()

    print("=== Knowledge Base System ===")
    print("Commands:")
    print("  add <file_path>     - Add a PDF, TXT, or CSV file")
    print("  list                - List all indexed documents")
    print("  ask <question>      - Ask a question")
    print("  clear               - Delete all documents")
    print("  quit                - Exit\n")

    while True:
        try:
            user_input = input("\nkb> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        parts = user_input.split(" ", 1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "quit":
            print("Goodbye!")
            break
        elif command == "add" and arg:
            kb.add_document(arg)
        elif command == "list":
            kb.list_sources()
        elif command == "ask" and arg:
            kb.query(arg)
        elif command == "clear":
            confirm = input("Are you sure? This deletes everything. (yes/no): ")
            if confirm.lower() == "yes":
                kb.clear()
        else:
            print("Unknown command. Try: add, list, ask, clear, quit")


if __name__ == "__main__":
    main()
```

#### How to Run

```bash
pip install langchain langchain-openai langchain-chroma chromadb pypdf

# Run the system
python knowledge_base.py
```

Example session:
```
kb> add ./reports/annual_report_2025.pdf
Loading: /path/to/reports/annual_report_2025.pdf
Split into 87 chunks
Added to knowledge base. Total chunks: 87

kb> add ./policies/remote_work.txt
Loading: /path/to/policies/remote_work.txt
Split into 12 chunks
Added to knowledge base. Total chunks: 99

kb> add ./data/employees.csv
Loading: /path/to/data/employees.csv
Split into 45 chunks
Added to knowledge base. Total chunks: 144

kb> list
Documents in knowledge base (3 files):
  - annual_report_2025.pdf
  - employees.csv
  - remote_work.txt
Total chunks: 144

kb> ask What was the total revenue in 2025?
Searching knowledge base for: 'What was the total revenue in 2025?'

Based on the annual report, total revenue for 2025 was $4.2 billion,
representing a 15% year-over-year increase [Source 1: annual_report_2025.pdf, Page: 12]...

kb> quit
```

#### Extensions
- Add a Streamlit or Flask web interface for browser-based access
- Support DOCX and Markdown files (add more loaders)
- Add metadata filtering (e.g., "search only in PDFs" or "search only in files added after a date")
- Implement document deletion by filename without clearing the entire database
- Add user authentication so multiple users can have separate knowledge bases
- Export Q&A pairs to create a FAQ document automatically

---

## Final Thoughts

RAG is arguably the most practical and impactful technique in applied GenAI today. It bridges the gap between a general-purpose LLM and a domain-specific expert system -- without the cost and complexity of fine-tuning. The key to building good RAG systems is not just the LLM or the prompt; it is the quality of your retrieval pipeline. Invest time in proper chunking, evaluate your retrieval results, and iterate.

Start with the code examples in this chapter. Build the PDF Chatbot project first -- it covers the entire pipeline end to end. Then move on to the Knowledge Base System to learn about multi-document management and citations. Once you are comfortable, explore advanced patterns like multi-query retrieval and hybrid search to push your RAG systems further.
