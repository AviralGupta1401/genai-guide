# Chapter 6: LangChain & LLM Application Development

---

## 1. Concept Explanation

### What Is LangChain?

LangChain is an open-source Python (and JavaScript) framework that makes it dramatically easier to build applications powered by large language models. If you have ever called the OpenAI or Anthropic API directly, you know you can send a prompt and get a response back. That works fine for a one-off question, but real-world applications need much more: they need to remember past conversations, pull in data from files or databases, chain multiple LLM calls together, and sometimes let the model decide which tool to use next. Wiring all of that up from scratch every single time is tedious and error-prone. LangChain gives you pre-built, composable pieces so you can focus on *what* your app does instead of *how* to glue everything together.

### The Car Analogy

Think of a large language model as a powerful engine. An engine by itself does not get you anywhere — you need a steering wheel, wheels, a fuel system, brakes, and a dashboard. LangChain is the car that wraps around the engine and connects all those parts. The engine (LLM) provides the raw intelligence; LangChain provides the structure that turns that intelligence into a usable application.

### Core Concepts at a Glance

| Concept | What It Does |
|---------|-------------|
| **Models** | Wrappers around LLM APIs (OpenAI, Anthropic, etc.) so you can swap providers with one line |
| **Prompts** | Templates that let you inject variables into prompts cleanly |
| **Chains** | Sequences of steps — take input, format a prompt, call an LLM, parse output |
| **Memory** | Stores conversation history so the LLM can "remember" what was said earlier |
| **Agents** | LLM-powered decision makers that choose which tools to call and in what order |
| **Tools** | Functions the agent can invoke — web search, calculator, database query, etc. |

---

## 2. Detailed Notes (Book Style)

### 2.1 LangChain Architecture

LangChain is organized into several packages:

- **`langchain-core`** — base abstractions (prompts, output parsers, runnables)
- **`langchain`** — chains, agents, and higher-level logic
- **`langchain-community`** — third-party integrations (vector stores, document loaders)
- **`langchain-openai`**, **`langchain-anthropic`** — provider-specific model wrappers
- **`langsmith`** — observability and tracing platform (optional but very useful)

### 2.2 LLM Wrappers

LangChain wraps every model provider behind a consistent interface. The two you will use most often:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

gpt = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
claude = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7)
```

Both expose the same `.invoke()`, `.stream()`, and `.batch()` methods. This means you can swap one model for another without changing the rest of your code.

### 2.3 Prompt Templates

Hard-coding prompts as f-strings is fragile. Prompt templates separate the *structure* of a prompt from the *data* that fills it.

**`PromptTemplate`** — for plain text prompts:

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    "Explain {topic} to a {audience} in {sentences} sentences."
)
```

**`ChatPromptTemplate`** — for chat-style prompts with roles:

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor who explains things simply."),
    ("human", "Explain {topic} in {sentences} sentences."),
])
```

### 2.4 Chains

A chain connects a prompt template to a model (and optionally an output parser). In modern LangChain, chains are built with **LCEL (LangChain Expression Language)** using the pipe operator (`|`):

```python
chain = prompt | model | output_parser
result = chain.invoke({"topic": "gravity", "sentences": "3"})
```

The older `LLMChain` class still works but is considered legacy. LCEL is the recommended approach.

**`SequentialChain`** — runs multiple chains in order, passing the output of one as input to the next. In LCEL, you simply pipe them together or use `RunnablePassthrough` to forward variables.

### 2.5 LCEL — LangChain Expression Language

LCEL is the modern way to compose LangChain components. Every component (prompt, model, parser, retriever) is a **Runnable** with three key methods:

| Method | Behavior |
|--------|----------|
| `.invoke(input)` | Process a single input synchronously |
| `.stream(input)` | Yield output chunks as they arrive |
| `.batch([inputs])` | Process multiple inputs in parallel |

The pipe operator creates a `RunnableSequence`:

```python
chain = prompt | model | parser
# Equivalent to: RunnableSequence(first=prompt, middle=[model], last=parser)
```

You can also use `RunnableParallel` to run branches side by side and `RunnablePassthrough` to forward inputs unchanged.

### 2.6 Output Parsers

Output parsers convert raw LLM text into structured data.

- **`StrOutputParser`** — extracts the string content from a chat message (the most common parser).
- **`JsonOutputParser`** — instructs the LLM to return JSON and parses it into a Python dict.
- **`PydanticOutputParser`** — validates output against a Pydantic model for type safety.

```python
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

str_parser = StrOutputParser()
json_parser = JsonOutputParser()
```

### 2.7 Memory Types

Memory lets a chatbot recall earlier parts of the conversation. LangChain offers several strategies:

| Memory Type | How It Works | Best For |
|-------------|-------------|----------|
| `ConversationBufferMemory` | Stores every message verbatim | Short conversations |
| `ConversationBufferWindowMemory` | Keeps only the last *k* exchanges | Medium conversations |
| `ConversationSummaryMemory` | Uses an LLM to summarize older messages | Long conversations |

Memory objects store messages and inject them into prompts automatically. In modern LCEL pipelines, you often manage history manually with a list of messages, but the memory classes remain useful for quick prototyping.

### 2.8 Document Loaders

Document loaders pull text from various sources into LangChain `Document` objects (which have `.page_content` and `.metadata`).

| Loader | Source |
|--------|--------|
| `TextLoader` | Plain `.txt` files |
| `PyPDFLoader` | PDF files (one document per page) |
| `CSVLoader` | CSV files (one document per row) |
| `WebBaseLoader` | Web pages |
| `DirectoryLoader` | All files in a folder |

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("notes.txt")
docs = loader.load()  # returns a list of Document objects
```

### 2.9 Text Splitters

LLMs have context-window limits. Text splitters break large documents into smaller, overlapping chunks so each chunk fits in the model's window while preserving context at the boundaries.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(docs)
```

`RecursiveCharacterTextSplitter` is the recommended default. It tries to split on paragraph boundaries first, then sentences, then words — preserving as much semantic coherence as possible.

---

## 3. Visual / Intuitive Explanation

### 3.1 A Chain as a Pipeline

Imagine an assembly line in a factory:

```
User Input
   |
   v
+---------------------+
| Prompt Template      |  <-- Slots the input into a well-structured prompt
+---------------------+
   |
   v
+---------------------+
| LLM (GPT / Claude)  |  <-- Generates a raw text response
+---------------------+
   |
   v
+---------------------+
| Output Parser        |  <-- Converts raw text into structured data
+---------------------+
   |
   v
Final Result (string, dict, Pydantic object, etc.)
```

Each box is a Runnable. The pipe operator (`|`) connects them left to right. Data flows through in sequence, and each step transforms it before handing it to the next.

### 3.2 How Memory Works

```
Turn 1:
  User: "My name is Aviral."
  Bot:  "Nice to meet you, Aviral!"
  --> Memory stores: [User: "My name is Aviral.", Bot: "Nice to meet you, Aviral!"]

Turn 2:
  User: "What is my name?"
  --> Memory injects stored messages into the prompt
  --> The LLM sees the full history and responds: "Your name is Aviral."
```

Without memory, every call to the LLM is independent — it would have no idea what "my name" refers to. Memory bridges turns by injecting prior messages into each new prompt.

### 3.3 Document Loading Pipeline

```
PDF / TXT / CSV file
       |
       v
  Document Loader      -->  List of Document objects
       |
       v
  Text Splitter         -->  Smaller chunks with overlap
       |
       v
  (Optional) Embeddings + Vector Store  -->  Semantic search index
       |
       v
  Retriever + LLM Chain -->  Answer user questions about the document
```

This is the backbone of every "chat with your documents" application. Chapters on RAG (Retrieval-Augmented Generation) build directly on this pipeline.

---

## 4. YouTube Resources

Search YouTube for the following to find high-quality tutorials:

- **"LangChain crash course 2024 Python"** — look for videos that cover LCEL, the modern API
- **"LangChain tutorial for beginners"** — step-by-step walkthroughs
- **"Build apps with LangChain Python"** — project-based tutorials
- **"LangChain Expression Language LCEL tutorial"** — focused on the pipe-operator style
- **"LangChain agents tools tutorial"** — how agents choose and invoke tools
- **"LangChain RAG tutorial"** — retrieval-augmented generation projects

Tip: Filter by upload date (last 12 months) since LangChain's API changes rapidly and older videos often use deprecated patterns.

---

## 5. Official Documentation

| Resource | URL | What to Read First |
|----------|-----|-------------------|
| LangChain Python Docs | https://python.langchain.com/docs | "Get Started" and "LCEL" sections |
| LangChain Cookbook | https://python.langchain.com/docs/use_cases | Practical recipes by use case |
| LangSmith Docs | https://docs.smith.langchain.com | "Quick Start" — learn to trace and debug chains |
| API Reference | https://api.python.langchain.com | Look up exact method signatures |
| LangChain GitHub | https://github.com/langchain-ai/langchain | Browse `/cookbook` and `/templates` folders |

**Recommended reading order:**

1. LCEL fundamentals (Runnables, pipe operator)
2. Chat models and prompt templates
3. Output parsers
4. Document loaders and text splitters
5. Memory / conversation history
6. Agents and tools

---

## 6. Code Examples

### 6.1 Installation

```bash
pip install langchain langchain-core langchain-openai langchain-anthropic
pip install langchain-community langchain-text-splitters
pip install python-dotenv
```

Set your API keys in a `.env` file:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 6.2 Basic Chain with LCEL (OpenAI)

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1. Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "Explain {topic} in {sentences} sentences."),
])

# 2. Choose a model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 3. Choose an output parser
parser = StrOutputParser()

# 4. Build the chain with the pipe operator
chain = prompt | model | parser

# 5. Run it
result = chain.invoke({"topic": "photosynthesis", "sentences": "3"})
print(result)
```

### 6.3 Same Chain with Claude (Anthropic)

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7)

chain = prompt | model | parser  # same prompt and parser, different model
result = chain.invoke({"topic": "photosynthesis", "sentences": "3"})
print(result)
```

### 6.4 Prompt Templates with Multiple Variables

```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
    """You are a {role}. A student asks: "{question}"
    
Respond in a {tone} tone, using at most {max_words} words."""
)

chain = template | model | StrOutputParser()
result = chain.invoke({
    "role": "physics professor",
    "question": "Why is the sky blue?",
    "tone": "friendly",
    "max_words": "100",
})
print(result)
```

### 6.5 Output Parsing to JSON

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You always respond in valid JSON."),
    ("human", 
     "List 3 facts about {topic}. "
     "Return a JSON object with a key 'facts' containing a list of strings."),
])

chain = prompt | model | JsonOutputParser()

result = chain.invoke({"topic": "Mars"})
print(type(result))  # <class 'dict'>
print(result["facts"])
```

### 6.6 Conversation with Memory

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | model | StrOutputParser()

# Maintain history as a simple list
history = []

def chat(user_message: str) -> str:
    response = chain.invoke({"input": user_message, "history": history})
    # Update history
    history.append(HumanMessage(content=user_message))
    history.append(AIMessage(content=response))
    return response

print(chat("Hi! My name is Aviral."))
print(chat("What is my name?"))         # The model remembers
print(chat("Tell me a joke about it.")) # Still has full context
```

### 6.7 Loading and Splitting Documents

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load
loader = TextLoader("my_notes.txt")
docs = loader.load()
print(f"Loaded {len(docs)} document(s), total chars: {len(docs[0].page_content)}")

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# Inspect the first chunk
print(chunks[0].page_content)
print(chunks[0].metadata)  # e.g., {'source': 'my_notes.txt'}
```

### 6.8 Sequential Chain (Multi-Step) with LCEL

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

# Step 1: Generate a story outline
outline_prompt = ChatPromptTemplate.from_messages([
    ("human", "Write a 3-point outline for a short story about {topic}."),
])

# Step 2: Write the story from the outline
story_prompt = ChatPromptTemplate.from_messages([
    ("human", "Using this outline, write a short story (200 words max):\n\n{outline}"),
])

# Compose them
outline_chain = outline_prompt | model | StrOutputParser()
story_chain = (
    {"outline": outline_chain}   # output of step 1 feeds into step 2
    | story_prompt
    | model
    | StrOutputParser()
)

result = story_chain.invoke({"topic": "a robot learning to paint"})
print(result)
```

---

## 7. Mini Practice Tasks

1. **Translator Chain** — Build a chain that takes text and a target language as input and returns the translated text. Test it with at least three languages.

2. **Multi-Model Comparison** — Write a script that sends the same prompt to both `ChatOpenAI` and `ChatAnthropic`, collects their responses, and prints them side by side. Use `RunnableParallel` if you can.

3. **Structured Output** — Create a chain that takes a movie title and returns a JSON object with keys `title`, `genre`, `year`, and `one_line_summary`. Validate that the result is a proper Python dict.

4. **Document Summarizer** — Load a `.txt` file, split it into chunks, and use a chain to summarize each chunk individually. Then combine the summaries into one final summary.

5. **Conversation Logger** — Build a chatbot with memory that also writes every exchange (user message + bot response) to a log file with timestamps.

---

## 8. Quick Revision Summary

- **LangChain** is a framework that connects LLMs to prompts, memory, data sources, and tools.
- **LCEL** (pipe operator `|`) is the modern way to compose chains: `prompt | model | parser`.
- **Prompt Templates** keep your prompts clean and reusable with variable placeholders.
- **Output Parsers** convert raw LLM text into strings, JSON dicts, or Pydantic objects.
- **Memory** injects conversation history into prompts so the LLM can recall past exchanges.
- **Document Loaders** pull text from files (TXT, PDF, CSV) into `Document` objects.
- **Text Splitters** break large documents into overlapping chunks that fit the model's context window.
- Every LCEL component is a **Runnable** with `.invoke()`, `.stream()`, and `.batch()` methods.
- Use **`langchain-openai`** and **`langchain-anthropic`** for model-specific wrappers; the interface is identical, so swapping models is trivial.
- **LangSmith** provides tracing and debugging — highly recommended during development.

---

## 9. Common Mistakes

### Mistake 1: Using Legacy `LLMChain` Instead of LCEL

```python
# OLD (legacy) - still works but discouraged
from langchain.chains import LLMChain
chain = LLMChain(llm=model, prompt=prompt)

# NEW (LCEL) - preferred
chain = prompt | model | StrOutputParser()
```

The legacy API will eventually be removed. Always use the pipe operator for new code.

### Mistake 2: Forgetting to Load Environment Variables

```python
# This will silently use an empty key and fail at runtime
model = ChatOpenAI(model="gpt-4o-mini")

# Fix: always load your .env file first
from dotenv import load_dotenv
load_dotenv()
```

### Mistake 3: Not Using an Output Parser

Without `StrOutputParser()`, the chain returns an `AIMessage` object, not a plain string. This trips up beginners who try to concatenate the result with other strings:

```python
# Returns AIMessage object - not what you usually want
chain = prompt | model
result = chain.invoke({"topic": "gravity"})
print(result)  # AIMessage(content="...", ...)

# Fix: add StrOutputParser
chain = prompt | model | StrOutputParser()
result = chain.invoke({"topic": "gravity"})
print(result)  # Just the text string
```

### Mistake 4: Context Window Overflow with Documents

Loading a 200-page PDF and stuffing every page into a single prompt will exceed the model's context window. Always split documents first:

```python
# Bad: passing all text at once
full_text = " ".join([doc.page_content for doc in docs])

# Good: split into chunks and process each one
chunks = splitter.split_documents(docs)
for chunk in chunks:
    result = chain.invoke({"text": chunk.page_content})
```

### Mistake 5: Confusing `invoke` with `run`

The `.run()` method is legacy. Always use `.invoke()` (single input), `.batch()` (multiple inputs), or `.stream()` (streaming output).

### Mistake 6: Forgetting `MessagesPlaceholder` for Chat History

If you want to inject conversation history, you must include a `MessagesPlaceholder` in your `ChatPromptTemplate`. Omitting it means the model never sees prior messages:

```python
# Wrong: no slot for history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("human", "{input}"),
])

# Correct: history placeholder included
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
```

### Mistake 7: Installing the Wrong Packages

LangChain is split across multiple packages. A common error:

```
ModuleNotFoundError: No module named 'langchain_openai'
```

Fix: install the provider-specific package, not just `langchain`:

```bash
pip install langchain-openai   # for ChatOpenAI
pip install langchain-anthropic # for ChatAnthropic
```

---

## Intermediate Projects

---

### Project 1: Quiz Generator

#### Problem Statement

Build a web application that generates multiple-choice quizzes on any topic. The user selects a topic, the number of questions, and a difficulty level. The app generates the quiz, lets the user answer, and shows their score.

#### Tech Stack

- Python 3.10+
- LangChain (LCEL chains, `JsonOutputParser`)
- Streamlit (web UI)
- OpenAI GPT-4o-mini or Claude

#### What You Will Learn

- Structured JSON output from an LLM
- Building interactive UIs with Streamlit
- Using LCEL chains with output parsing
- Session state management in Streamlit

#### Full Code

**`quiz_generator.py`**

```python
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# --- LangChain Setup ---
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

quiz_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a quiz generator. Always return valid JSON and nothing else."),
    ("human",
     """Generate a multiple-choice quiz with the following parameters:
- Topic: {topic}
- Number of questions: {num_questions}
- Difficulty: {difficulty}

Return a JSON object with this exact structure:
{{
  "quiz_title": "string",
  "questions": [
    {{
      "question": "string",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "A",
      "explanation": "string"
    }}
  ]
}}"""),
])

quiz_chain = quiz_prompt | model | JsonOutputParser()

# --- Streamlit UI ---
st.set_page_config(page_title="Quiz Generator", page_icon="📝")
st.title("Quiz Generator")

# Sidebar controls
with st.sidebar:
    topic = st.text_input("Topic", value="Python programming")
    num_questions = st.slider("Number of questions", 3, 10, 5)
    difficulty = st.select_slider(
        "Difficulty", options=["Easy", "Medium", "Hard"], value="Medium"
    )
    generate = st.button("Generate Quiz", type="primary")

# Initialize session state
if "quiz" not in st.session_state:
    st.session_state.quiz = None
    st.session_state.answers = {}
    st.session_state.submitted = False

# Generate quiz
if generate:
    with st.spinner("Generating quiz..."):
        st.session_state.quiz = quiz_chain.invoke({
            "topic": topic,
            "num_questions": str(num_questions),
            "difficulty": difficulty,
        })
        st.session_state.answers = {}
        st.session_state.submitted = False

# Display quiz
if st.session_state.quiz:
    quiz = st.session_state.quiz
    st.header(quiz.get("quiz_title", "Quiz"))

    for i, q in enumerate(quiz["questions"]):
        st.subheader(f"Question {i + 1}")
        st.write(q["question"])

        # Extract just the letter labels
        option_labels = [opt[0] for opt in q["options"]]
        st.session_state.answers[i] = st.radio(
            "Your answer:",
            q["options"],
            key=f"q_{i}",
            label_visibility="collapsed",
        )

    if st.button("Submit Answers"):
        st.session_state.submitted = True

    if st.session_state.submitted:
        score = 0
        st.divider()
        st.header("Results")
        for i, q in enumerate(quiz["questions"]):
            user_ans = st.session_state.answers.get(i, "")
            correct_letter = q["correct_answer"]
            user_letter = user_ans[0] if user_ans else ""

            if user_letter == correct_letter:
                score += 1
                st.success(f"Q{i+1}: Correct!")
            else:
                st.error(f"Q{i+1}: Wrong. Correct answer: {correct_letter}")
            st.caption(f"Explanation: {q['explanation']}")

        st.divider()
        pct = int(score / len(quiz["questions"]) * 100)
        st.metric("Score", f"{score}/{len(quiz['questions'])} ({pct}%)")
```

#### How to Run

```bash
pip install streamlit langchain langchain-openai python-dotenv
# Create a .env file with OPENAI_API_KEY=sk-...
streamlit run quiz_generator.py
```

#### Extensions

- Add a timer for each question.
- Support true/false and fill-in-the-blank formats.
- Store quiz results in a database and track progress over time.
- Let users paste their own study material and generate quizzes from it.

---

### Project 2: Document Q&A System

#### Problem Statement

Build an application where a user uploads a text file and then asks questions about its content. The system loads the document, splits it into chunks, and uses the most relevant chunks to answer each question.

#### Tech Stack

- Python 3.10+
- LangChain (document loaders, text splitters, LCEL chains)
- Streamlit (web UI)
- OpenAI GPT-4o-mini

#### What You Will Learn

- Loading and chunking documents with LangChain
- Simple keyword-based retrieval (no vector store needed for this project)
- Stuffing context into prompts
- Building a document-aware Q&A chain

#### Full Code

**`doc_qa.py`**

```python
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful assistant that answers questions based on the 
provided context. If the answer is not in the context, say "I cannot find 
the answer in the provided document." Do not make up information."""),
    ("human",
     """Context:
{context}

Question: {question}

Answer the question based only on the context above."""),
])

qa_chain = qa_prompt | model | StrOutputParser()

# --- Helper Functions ---

def load_and_split(file_path: str, chunk_size: int = 800, overlap: int = 100):
    """Load a text file and split it into chunks."""
    loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_documents(docs)


def find_relevant_chunks(chunks, question: str, top_k: int = 3):
    """Simple keyword-based relevance scoring (no vector store needed).
    
    For production use, replace this with a proper vector store 
    (FAISS, Chroma, etc.) and embedding-based retrieval.
    """
    question_words = set(question.lower().split())
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.page_content.lower().split())
        overlap = len(question_words & chunk_words)
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


# --- Streamlit UI ---

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("Document Q&A System")
st.caption("Upload a text file and ask questions about it.")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Load and split
    if "chunks" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Processing document..."):
            st.session_state.chunks = load_and_split(tmp_path)
            st.session_state.file_name = uploaded_file.name
        st.success(
            f"Loaded **{uploaded_file.name}** — "
            f"split into {len(st.session_state.chunks)} chunks."
        )

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User input
    if question := st.chat_input("Ask a question about the document"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Find relevant chunks and generate answer
        relevant = find_relevant_chunks(st.session_state.chunks, question)
        context = "\n\n---\n\n".join([c.page_content for c in relevant])

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = qa_chain.invoke({
                    "context": context,
                    "question": question,
                })
                st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sidebar: show document stats
    with st.sidebar:
        st.subheader("Document Info")
        st.write(f"**File:** {uploaded_file.name}")
        st.write(f"**Chunks:** {len(st.session_state.chunks)}")
        st.write(f"**Total characters:** "
                 f"{sum(len(c.page_content) for c in st.session_state.chunks):,}")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Clean up temp file
    os.unlink(tmp_path)
```

#### How to Run

```bash
pip install streamlit langchain langchain-openai langchain-community \
    langchain-text-splitters python-dotenv
streamlit run doc_qa.py
```

Upload any `.txt` file, then ask questions in the chat interface. The system finds the most relevant chunks and sends them as context to the LLM.

#### Extensions

- Add PDF support with `PyPDFLoader`.
- Replace keyword search with proper vector-store retrieval (FAISS or Chroma) for much better accuracy.
- Support multiple file uploads.
- Show which chunks were used to answer each question (source attribution).

---

### Project 3: Chatbot with Memory

#### Problem Statement

Build a conversational chatbot that remembers everything said in the conversation. The user can ask the bot to recall earlier messages. The project demonstrates three different memory strategies so you can compare their behavior.

#### Tech Stack

- Python 3.10+
- LangChain (LCEL chains, message history management)
- Streamlit (web UI)
- Anthropic Claude (to show a non-OpenAI example)

#### What You Will Learn

- Managing conversation history with LangChain message types
- Implementing buffer, window, and summary memory from scratch
- Streaming responses in Streamlit
- Using Claude via `langchain-anthropic`

#### Full Code

**`chatbot_memory.py`**

```python
import streamlit as st
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# --- Memory Strategies ---

class BufferMemory:
    """Stores all messages. Simple but grows without limit."""
    def __init__(self):
        self.messages = []

    def add_user(self, text):
        self.messages.append(HumanMessage(content=text))

    def add_ai(self, text):
        self.messages.append(AIMessage(content=text))

    def get_messages(self):
        return self.messages.copy()

    def clear(self):
        self.messages = []

    def info(self):
        return f"Storing all {len(self.messages)} messages"


class WindowMemory:
    """Keeps only the last k exchanges (2k messages)."""
    def __init__(self, k=5):
        self.messages = []
        self.k = k

    def add_user(self, text):
        self.messages.append(HumanMessage(content=text))

    def add_ai(self, text):
        self.messages.append(AIMessage(content=text))
        # Trim to last k exchanges (each exchange = 2 messages)
        max_messages = self.k * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self):
        return self.messages.copy()

    def clear(self):
        self.messages = []

    def info(self):
        return (f"Window of last {self.k} exchanges "
                f"({len(self.messages)} messages stored)")


class SummaryMemory:
    """Summarizes older messages to save tokens."""
    def __init__(self, llm, threshold=6):
        self.messages = []
        self.summary = ""
        self.llm = llm
        self.threshold = threshold  # summarize when messages exceed this

    def add_user(self, text):
        self.messages.append(HumanMessage(content=text))

    def add_ai(self, text):
        self.messages.append(AIMessage(content=text))
        if len(self.messages) > self.threshold:
            self._summarize()

    def _summarize(self):
        """Summarize older messages and keep only recent ones."""
        to_summarize = self.messages[:-4]  # keep last 2 exchanges
        recent = self.messages[-4:]

        conversation_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in to_summarize
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this conversation concisely, "
                       "preserving all key facts and names mentioned."),
            ("human", 
             "Previous summary: {prev_summary}\n\n"
             "New messages:\n{conversation}\n\n"
             "Updated summary:"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        self.summary = chain.invoke({
            "prev_summary": self.summary or "(none)",
            "conversation": conversation_text,
        })
        self.messages = recent

    def get_messages(self):
        msgs = []
        if self.summary:
            msgs.append(SystemMessage(
                content=f"Summary of earlier conversation: {self.summary}"
            ))
        msgs.extend(self.messages)
        return msgs

    def clear(self):
        self.messages = []
        self.summary = ""

    def info(self):
        status = f"{len(self.messages)} recent messages"
        if self.summary:
            status += f" + summary ({len(self.summary)} chars)"
        return status


# --- Streamlit UI ---

st.set_page_config(page_title="Chatbot with Memory", page_icon="🤖")
st.title("Chatbot with Memory")

# Sidebar: configuration
with st.sidebar:
    st.subheader("Settings")
    
    model_choice = st.selectbox("Model", ["Claude (Anthropic)", "GPT (OpenAI)"])
    memory_type = st.selectbox(
        "Memory Type",
        ["Buffer (full history)", "Window (last k)", "Summary (compressed)"]
    )
    
    if "Window" in memory_type:
        window_k = st.slider("Window size (k exchanges)", 2, 10, 5)
    
    if st.button("Reset Conversation"):
        st.session_state.pop("memory", None)
        st.session_state.pop("chat_messages", None)
        st.rerun()

# Initialize model
if "Claude" in model_choice:
    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0.7)
else:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Initialize memory
if "memory" not in st.session_state:
    if "Buffer" in memory_type:
        st.session_state.memory = BufferMemory()
    elif "Window" in memory_type:
        st.session_state.memory = WindowMemory(k=window_k)
    else:
        st.session_state.memory = SummaryMemory(llm=llm)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

memory = st.session_state.memory

# Build chain
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly and helpful assistant. You remember everything "
     "the user has told you in this conversation. If they ask you to recall "
     "something, do your best using the conversation history."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm | StrOutputParser()

# Display chat history
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if user_input := st.chat_input("Say something..."):
    # Show user message
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({
                "input": user_input,
                "history": memory.get_messages(),
            })
            st.write(response)

    # Update memory
    memory.add_user(user_input)
    memory.add_ai(response)
    st.session_state.chat_messages.append({"role": "assistant", "content": response})

# Sidebar: memory info
with st.sidebar:
    st.divider()
    st.subheader("Memory Status")
    st.caption(memory.info())
```

#### How to Run

```bash
pip install streamlit langchain langchain-openai langchain-anthropic python-dotenv
# Create a .env file with your API keys
streamlit run chatbot_memory.py
```

Try telling the chatbot your name, favorite color, or a story. Then ask it to recall those details. Switch between memory types to see how each one behaves differently:

- **Buffer** remembers everything (but token usage grows).
- **Window** forgets older messages (but keeps token usage constant).
- **Summary** compresses old messages into a summary (balancing recall and cost).

#### Extensions

- Add a "memory inspector" panel that shows exactly what the LLM sees at each turn.
- Implement a hybrid memory that uses window for recent messages and summary for older ones.
- Add the ability to export/import conversation history as JSON.
- Integrate tools (web search, calculator) to make it an agent with memory.
- Add token counting to show the user how many tokens each memory strategy uses per turn.

---

*End of Chapter 6. In the next chapter, we will explore vector databases and embeddings, which power the retrieval step in RAG (Retrieval-Augmented Generation) applications.*
