# Chapter 10: Final Projects & Interview Preparation

This capstone chapter brings together everything you have learned. Part 1 presents three resume-worthy projects you can build, deploy, and discuss in interviews. Part 2 arms you with 50 interview questions, a structured method for talking about your projects, and a cheat sheet of key terms.

---

# PART 1: Resume-Level Projects

---

## Project 1: Multi-Document AI Research Assistant

### Problem Statement

Researchers, lawyers, and analysts routinely work with dozens of PDFs and documents. They need to ask questions that span multiple files and get answers with precise source citations. Build a Streamlit application that lets users upload multiple documents, indexes them into a vector store, and provides a conversational Q&A interface with memory and source attribution.

### Tech Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3.10+ |
| LLM Orchestration | LangChain |
| Vector Database | ChromaDB |
| LLM | OpenAI GPT-4o (or Claude via Anthropic SDK) |
| Embeddings | OpenAI `text-embedding-3-small` |
| PDF Parsing | PyPDFLoader |
| Web UI | Streamlit |
| Environment | python-dotenv |

### Features

- Upload multiple PDFs and text files in a single session
- Automatic chunking, embedding, and indexing into ChromaDB
- Conversational Q&A with memory across turns
- Source citations showing filename and page number for every answer
- Export conversation history to a text file
- Persistent vector store so documents survive app restarts

### Architecture

1. The user opens the Streamlit app and uploads one or more PDF or text files through the sidebar.
2. Each file is read with the appropriate LangChain document loader (PyPDFLoader for PDFs, TextLoader for .txt files). The loader attaches metadata including filename and page number.
3. The loaded documents are split into chunks of approximately 1000 characters with 200 characters of overlap using RecursiveCharacterTextSplitter.
4. Each chunk is embedded using OpenAI embeddings and stored in a ChromaDB collection on disk.
5. When the user types a question, the app retrieves the top 5 most relevant chunks from ChromaDB using similarity search.
6. The retrieved chunks, together with conversation history, are formatted into a prompt and sent to the LLM.
7. The LLM generates an answer. The app extracts source metadata from the retrieved documents and displays them as citations below the answer.
8. Conversation history is maintained in Streamlit session state so follow-up questions work naturally.

### Why It Is Industry-Relevant

Enterprise knowledge management is a multi-billion dollar market. Law firms use document Q&A to review contracts. Research teams use it to synthesize literature. Consulting firms use it to mine past deliverables. This project demonstrates RAG, vector databases, document processing, and conversational UX -- the exact stack companies are hiring for today.

### Complete Code

```python
# research_assistant.py

import os
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()

CHROMA_DIR = "./chroma_research_db"
COLLECTION = "research_docs"


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=get_embeddings(),
    )


def process_uploaded_files(uploaded_files):
    """Load, chunk, and index uploaded files into ChromaDB."""
    all_docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if suffix.lower() == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        docs = loader.load()
        for doc in docs:
            doc.metadata["source_filename"] = uploaded_file.name
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)
        os.unlink(tmp_path)

    if all_docs:
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=get_embeddings(),
            collection_name=COLLECTION,
            persist_directory=CHROMA_DIR,
        )
        return vectorstore, len(all_docs)
    return None, 0


def build_chain(vectorstore):
    """Build a conversational retrieval chain with memory."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    memory = ConversationBufferWindowMemory(
        k=8,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=True,
    )
    return chain


def format_sources(source_docs):
    """Extract unique sources from retrieved documents."""
    seen = set()
    sources = []
    for doc in source_docs:
        filename = doc.metadata.get("source_filename", "unknown")
        page = doc.metadata.get("page", "N/A")
        key = f"{filename}::page {page}"
        if key not in seen:
            seen.add(key)
            sources.append(f"- **{filename}**, page {page}")
    return "\n".join(sources) if sources else "No sources found."


def export_history(messages):
    """Format conversation history as plain text for download."""
    lines = [f"Research Assistant - Conversation Export ({datetime.now():%Y-%m-%d %H:%M})\n"]
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}\n")
    return "\n".join(lines)


# ---- Streamlit UI ----

st.set_page_config(page_title="Multi-Doc Research Assistant", layout="wide")
st.title("Multi-Document AI Research Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None

# Sidebar: file upload and controls
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Index Documents") and uploaded_files:
        with st.spinner("Processing and indexing documents..."):
            vectorstore, num_chunks = process_uploaded_files(uploaded_files)
            if vectorstore:
                st.session_state.chain = build_chain(vectorstore)
                st.success(f"Indexed {num_chunks} chunks from {len(uploaded_files)} file(s).")

    # Load existing DB if present
    if st.session_state.chain is None and os.path.exists(CHROMA_DIR):
        try:
            vs = get_vectorstore()
            if vs._collection.count() > 0:
                st.session_state.chain = build_chain(vs)
                st.info("Loaded existing knowledge base from disk.")
        except Exception:
            pass

    st.divider()
    if st.session_state.messages:
        st.download_button(
            "Export Conversation",
            data=export_history(st.session_state.messages),
            file_name="conversation_export.txt",
        )
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.chain is None:
        st.warning("Please upload and index documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = st.session_state.chain.invoke({"question": prompt})
                answer = result["answer"]
                sources = format_sources(result.get("source_documents", []))
                full_response = f"{answer}\n\n---\n**Sources:**\n{sources}"
                st.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
```

### How to Run

```bash
pip install streamlit langchain langchain-community langchain-openai chromadb pypdf python-dotenv
export OPENAI_API_KEY="sk-..."
streamlit run research_assistant.py
```

### How to Extend It

- Add support for Word documents (.docx) and HTML files using LangChain's respective loaders.
- Add a "summarize all documents" button that generates a high-level summary of the entire knowledge base.
- Implement user authentication so multiple users can maintain separate knowledge bases.
- Replace OpenAI with a local model (Ollama + Llama 3) for fully offline operation.
- Add metadata filtering so users can restrict searches to specific files or date ranges.

---

## Project 2: AI-Powered Customer Support Bot

### Problem Statement

Every company with a product needs customer support. Build an intelligent support chatbot that answers questions from company documentation using RAG, detects customer frustration through sentiment analysis, and escalates to a human agent when needed. The system exposes both a chat UI and a REST API.

### Tech Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3.10+ |
| LLM Orchestration | LangChain |
| Vector Database | ChromaDB |
| LLM | OpenAI GPT-4o |
| Embeddings | OpenAI `text-embedding-3-small` |
| Sentiment Analysis | TextBlob |
| REST API | FastAPI + Uvicorn |
| Web UI | Streamlit |
| Environment | python-dotenv |

### Features

- RAG-based answers grounded in company documentation
- Sentiment analysis on every user message with automatic escalation on negative sentiment
- Conversation history maintained per session
- Confidence-based fallback: if retrieved chunks have low relevance, the bot admits it does not know
- REST API endpoint for integration with existing systems
- Admin panel concept: view recent conversations and flag escalated ones

### Architecture

1. Company FAQ and documentation files are loaded at startup, chunked, embedded, and stored in ChromaDB.
2. A user sends a message through the Streamlit UI or the FastAPI endpoint.
3. The system runs sentiment analysis on the user message using TextBlob. If the polarity score is below -0.3, the message is flagged for escalation.
4. The user question is used to retrieve the top 4 most relevant chunks from ChromaDB.
5. The system checks the relevance scores of the retrieved chunks. If the best score is below a threshold, the bot responds with a fallback message offering to connect the user with a human agent.
6. Otherwise, the retrieved context and conversation history are sent to the LLM, which generates a grounded answer.
7. The response, along with sentiment and escalation status, is returned to the user.
8. All conversations are logged in memory for the admin view.

### Why It Is Industry-Relevant

Customer support automation is one of the most common GenAI deployments. Companies from startups to Fortune 500 are building exactly this kind of system. The combination of RAG, sentiment detection, and escalation logic shows practical engineering judgment, not just API calls.

### Complete Code

```python
# support_bot.py

import os
import json
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from textblob import TextBlob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

load_dotenv()

CHROMA_DIR = "./chroma_support_db"
COLLECTION = "support_docs"

# ---------- Sample FAQ data (replace with real docs) ----------

SAMPLE_FAQ = [
    {
        "question": "How do I reset my password?",
        "answer": "Go to Settings > Account > Reset Password. Click the reset link sent to your email. The link expires in 24 hours."
    },
    {
        "question": "What are the pricing plans?",
        "answer": "We offer three plans: Starter at $9/month (1 user, 1GB storage), Pro at $29/month (5 users, 10GB storage), and Enterprise at custom pricing (unlimited users, unlimited storage, dedicated support)."
    },
    {
        "question": "How do I cancel my subscription?",
        "answer": "Go to Settings > Billing > Cancel Subscription. Your access continues until the end of the current billing period. Refunds are available within the first 14 days."
    },
    {
        "question": "What file formats are supported?",
        "answer": "We support PDF, DOCX, TXT, CSV, XLSX, PNG, JPG, and GIF. Maximum file size is 50MB per file, 500MB total storage on the Starter plan."
    },
    {
        "question": "How do I contact human support?",
        "answer": "Email support@example.com, use the live chat during business hours (9 AM - 6 PM EST), or call 1-800-EXAMPLE. Enterprise customers have a dedicated Slack channel."
    },
    {
        "question": "Is my data secure?",
        "answer": "Yes. We use AES-256 encryption at rest, TLS 1.3 in transit, and are SOC 2 Type II certified. Data is stored in AWS US regions. We never share data with third parties."
    },
    {
        "question": "How do integrations work?",
        "answer": "We integrate with Slack, Jira, GitHub, Google Drive, and Zapier. Go to Settings > Integrations to connect. API access is available on Pro and Enterprise plans."
    },
]


def build_knowledge_base():
    """Index FAQ data into ChromaDB."""
    docs = []
    for item in SAMPLE_FAQ:
        content = f"Question: {item['question']}\nAnswer: {item['answer']}"
        docs.append(Document(page_content=content, metadata={"source": "FAQ"}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    return vectorstore


def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    )


def analyze_sentiment(text):
    """Return sentiment polarity and label."""
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.3:
        return polarity, "negative"
    elif polarity > 0.3:
        return polarity, "positive"
    return polarity, "neutral"


def should_escalate(sentiment_label, message_text):
    """Decide if a conversation should be escalated to a human."""
    escalation_keywords = ["speak to human", "talk to agent", "real person", "manager", "complaint"]
    if sentiment_label == "negative":
        return True
    if any(kw in message_text.lower() for kw in escalation_keywords):
        return True
    return False


SYSTEM_PROMPT = """You are a helpful customer support assistant for a SaaS company.
Answer questions based ONLY on the provided context. If the context does not contain
enough information to answer, say: "I don't have enough information to answer that.
Let me connect you with a human agent."

Be concise, friendly, and professional. If the user seems frustrated, acknowledge
their frustration before answering.

Context:
{context}

Conversation history:
{history}
"""


def generate_response(question, context_docs, chat_history):
    """Generate a response using the LLM with retrieved context."""
    context = "\n\n".join([doc.page_content for doc in context_docs])
    history = "\n".join(
        [f"{msg['role'].title()}: {msg['content']}" for msg in chat_history[-6:]]
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    chain = prompt | llm
    result = chain.invoke({
        "context": context,
        "history": history,
        "question": question,
    })
    return result.content


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Customer Support Bot", layout="wide")

tab_chat, tab_admin = st.tabs(["Chat Support", "Admin Panel"])

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "escalated" not in st.session_state:
    st.session_state.escalated = False

# Initialize knowledge base
if not os.path.exists(CHROMA_DIR):
    with st.spinner("Building knowledge base from FAQ..."):
        build_knowledge_base()

vectorstore = get_vectorstore()

with tab_chat:
    st.title("Customer Support")
    st.caption("Ask me anything about our product. I can also connect you with a human agent.")

    if st.session_state.escalated:
        st.warning("This conversation has been flagged for human review. A support agent will follow up shortly.")

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sentiment"):
                sentiment_color = {
                    "positive": "green", "neutral": "gray", "negative": "red"
                }
                color = sentiment_color.get(msg["sentiment"], "gray")
                st.caption(f"Sentiment: :{color}[{msg['sentiment']}]")

    if prompt := st.chat_input("Type your question here..."):
        polarity, sentiment_label = analyze_sentiment(prompt)
        user_msg = {"role": "user", "content": prompt, "sentiment": sentiment_label}
        st.session_state.chat_messages.append(user_msg)

        with st.chat_message("user"):
            st.markdown(prompt)

        escalate = should_escalate(sentiment_label, prompt)
        if escalate:
            st.session_state.escalated = True

        with st.chat_message("assistant"):
            with st.spinner("Looking up your answer..."):
                results = vectorstore.similarity_search_with_relevance_scores(prompt, k=4)
                context_docs = [doc for doc, score in results if score > 0.3]

                if not context_docs:
                    answer = (
                        "I don't have specific information about that in our documentation. "
                        "Let me connect you with a human agent who can help. You can also "
                        "email support@example.com or call 1-800-EXAMPLE."
                    )
                elif escalate:
                    answer = generate_response(prompt, context_docs, st.session_state.chat_messages)
                    answer += (
                        "\n\n---\n*I've also flagged this conversation for a human agent "
                        "to follow up with you shortly.*"
                    )
                else:
                    answer = generate_response(prompt, context_docs, st.session_state.chat_messages)

                st.markdown(answer)

        assistant_msg = {"role": "assistant", "content": answer}
        st.session_state.chat_messages.append(assistant_msg)

        # Log conversation for admin
        st.session_state.conversation_log.append({
            "session_id": st.session_state.session_id,
            "timestamp": datetime.now().isoformat(),
            "user_message": prompt,
            "sentiment": sentiment_label,
            "escalated": escalate,
            "response": answer,
        })

with tab_admin:
    st.title("Admin Panel")
    st.caption("Review recent conversations and escalations.")

    logs = st.session_state.conversation_log
    if not logs:
        st.info("No conversations yet.")
    else:
        escalated_logs = [l for l in logs if l["escalated"]]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Messages", len(logs))
        col2.metric("Escalated", len(escalated_logs))
        col3.metric("Escalation Rate", f"{len(escalated_logs)/len(logs)*100:.0f}%")

        st.subheader("Escalated Conversations")
        if escalated_logs:
            for log in escalated_logs:
                with st.expander(f"Session {log['session_id']} - {log['timestamp'][:16]}"):
                    st.write(f"**User:** {log['user_message']}")
                    st.write(f"**Sentiment:** {log['sentiment']}")
                    st.write(f"**Response:** {log['response'][:200]}...")
        else:
            st.success("No escalations.")

        st.subheader("All Recent Messages")
        for log in reversed(logs[-20:]):
            flag = " ESCALATED" if log["escalated"] else ""
            st.text(f"[{log['timestamp'][:16]}] [{log['sentiment']}]{flag} {log['user_message'][:80]}")
```

### How to Run

```bash
pip install streamlit langchain langchain-community langchain-openai chromadb textblob python-dotenv
python -m textblob.download_corpora
export OPENAI_API_KEY="sk-..."
streamlit run support_bot.py
```

### How to Extend It

- Replace sample FAQ with a loader that reads from a docs folder, Notion export, or Confluence API.
- Add a FastAPI backend so the bot can be embedded in any website via a REST endpoint.
- Integrate with Slack or Microsoft Teams for direct messaging support.
- Store conversation logs in a database (SQLite or PostgreSQL) for persistence and analytics.
- Add multilingual support by detecting the user's language and responding accordingly.

---

## Project 3: Intelligent Content Generator Platform

### Problem Statement

Marketing teams spend hours producing blog posts, social media captions, and email campaigns. Build a platform that generates multiple content types with consistent brand voice, adjustable tone, and SEO awareness.

### Tech Stack

| Component | Library / Tool |
|---|---|
| Language | Python 3.10+ |
| LLM Orchestration | LangChain |
| LLM | OpenAI GPT-4o |
| Web UI | Streamlit |
| Environment | python-dotenv |

### Features

- Generate blog posts, social media posts (Twitter/LinkedIn/Instagram), and marketing emails
- Tone and style selection (professional, casual, witty, inspirational)
- Customizable brand voice description
- Content templates with structured output
- SEO keyword suggestions for blog content
- Word count and character count controls
- One-click copy and export

### Architecture

1. The user selects a content type (blog post, social media, or email) from the sidebar.
2. The user fills in the topic, target audience, desired tone, and optional brand voice description.
3. For blog posts, the user can also enter target SEO keywords.
4. The app constructs a specialized prompt using LangChain prompt templates tailored to the selected content type.
5. The prompt is sent to the LLM, which generates structured content (title, body, hashtags, subject line, etc., depending on type).
6. The generated content is displayed in an editable text area so the user can tweak it.
7. For blog posts, a second LLM call generates SEO suggestions (meta description, keyword density tips, heading recommendations).
8. The user can regenerate, adjust tone, or export the content.

### Why It Is Industry-Relevant

Content marketing is a $400+ billion industry. Agencies and in-house teams are adopting AI tools to accelerate production without sacrificing quality. This project demonstrates prompt engineering, structured output, and user-facing product design -- skills every GenAI engineer needs.

### Complete Code

```python
# content_generator.py

import os
import json

import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
llm_precise = ChatOpenAI(model="gpt-4o", temperature=0.2)


TONES = ["Professional", "Casual", "Witty", "Inspirational", "Authoritative", "Friendly"]

# ---------- Prompt Templates ----------

BLOG_PROMPT = """You are an expert content writer.
Write a blog post on the following topic.

Topic: {topic}
Target Audience: {audience}
Tone: {tone}
Approximate Word Count: {word_count}
{brand_voice_instruction}
{seo_instruction}

Structure the blog post with:
1. An engaging title
2. An introduction that hooks the reader
3. 3-5 subheadings with detailed content under each
4. A conclusion with a call to action

Write in Markdown format."""

SOCIAL_PROMPT = """You are a social media content expert.
Create {platform} posts about the following topic.

Topic: {topic}
Target Audience: {audience}
Tone: {tone}
{brand_voice_instruction}

Generate 3 different post variations. For each post:
- Keep within {platform}'s best practices for length
- Include relevant hashtags
- Include an emoji strategy appropriate for the platform
- End with a call to action when appropriate

Format each variation clearly with "Variation 1:", "Variation 2:", "Variation 3:" headers."""

EMAIL_PROMPT = """You are an expert email copywriter.
Write a marketing email on the following topic.

Topic: {topic}
Target Audience: {audience}
Tone: {tone}
Email Type: {email_type}
{brand_voice_instruction}

Provide:
1. **Subject Line** (3 options, each under 60 characters)
2. **Preview Text** (under 100 characters)
3. **Email Body** with:
   - A compelling opening line
   - The main message (2-3 short paragraphs)
   - A clear call-to-action button text
   - A sign-off
4. **Alternative Subject Lines** for A/B testing (2 more options)"""

SEO_PROMPT = """Analyze the following blog post and provide SEO recommendations.

Blog Post:
{content}

Target Keywords: {keywords}

Provide:
1. **Meta Description** (under 160 characters)
2. **Keyword Analysis**: Are the target keywords naturally included? Suggest placements if not.
3. **Heading Optimization**: Suggest improvements to subheadings for SEO.
4. **Internal Linking Suggestions**: What related topics could be linked?
5. **Readability Score**: Estimate and suggest improvements.

Be specific and actionable."""


def generate_blog(topic, audience, tone, word_count, brand_voice, keywords):
    brand_instruction = f"Brand Voice: {brand_voice}" if brand_voice else ""
    seo_instruction = f"Target SEO Keywords: {keywords}" if keywords else ""

    prompt = ChatPromptTemplate.from_template(BLOG_PROMPT)
    chain = prompt | llm
    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "tone": tone,
        "word_count": word_count,
        "brand_voice_instruction": brand_instruction,
        "seo_instruction": seo_instruction,
    })
    return result.content


def generate_social(topic, audience, tone, platform, brand_voice):
    brand_instruction = f"Brand Voice: {brand_voice}" if brand_voice else ""

    prompt = ChatPromptTemplate.from_template(SOCIAL_PROMPT)
    chain = prompt | llm
    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "tone": tone,
        "platform": platform,
        "brand_voice_instruction": brand_instruction,
    })
    return result.content


def generate_email(topic, audience, tone, email_type, brand_voice):
    brand_instruction = f"Brand Voice: {brand_voice}" if brand_voice else ""

    prompt = ChatPromptTemplate.from_template(EMAIL_PROMPT)
    chain = prompt | llm
    result = chain.invoke({
        "topic": topic,
        "audience": audience,
        "tone": tone,
        "email_type": email_type,
        "brand_voice_instruction": brand_instruction,
    })
    return result.content


def generate_seo_suggestions(content, keywords):
    prompt = ChatPromptTemplate.from_template(SEO_PROMPT)
    chain = prompt | llm_precise
    result = chain.invoke({"content": content, "keywords": keywords})
    return result.content


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Content Generator", layout="wide")
st.title("Intelligent Content Generator Platform")

with st.sidebar:
    st.header("Content Settings")

    content_type = st.selectbox("Content Type", ["Blog Post", "Social Media", "Marketing Email"])
    topic = st.text_input("Topic / Subject", placeholder="e.g., Benefits of remote work")
    audience = st.text_input("Target Audience", placeholder="e.g., HR managers at mid-size companies")
    tone = st.selectbox("Tone", TONES)
    brand_voice = st.text_area(
        "Brand Voice (optional)",
        placeholder="e.g., We are a modern fintech startup. We speak plainly, avoid jargon, and use data to back claims.",
        height=80,
    )

    # Type-specific options
    if content_type == "Blog Post":
        word_count = st.slider("Target Word Count", 300, 2000, 800, 100)
        keywords = st.text_input("SEO Keywords (comma-separated)", placeholder="e.g., remote work, productivity, hybrid")
    elif content_type == "Social Media":
        platform = st.selectbox("Platform", ["Twitter/X", "LinkedIn", "Instagram"])
    elif content_type == "Marketing Email":
        email_type = st.selectbox("Email Type", [
            "Product Launch", "Newsletter", "Promotional Offer",
            "Welcome Email", "Re-engagement",
        ])

    generate_btn = st.button("Generate Content", type="primary", use_container_width=True)

# Main area
if generate_btn and topic and audience:
    with st.spinner("Generating content..."):
        if content_type == "Blog Post":
            content = generate_blog(topic, audience, tone, word_count, brand_voice, keywords)

            st.subheader("Generated Blog Post")
            st.markdown(content)

            st.divider()
            edited = st.text_area("Edit Content", value=content, height=400)

            if keywords:
                st.divider()
                with st.spinner("Analyzing SEO..."):
                    seo = generate_seo_suggestions(content, keywords)
                st.subheader("SEO Recommendations")
                st.markdown(seo)

        elif content_type == "Social Media":
            content = generate_social(topic, audience, tone, platform, brand_voice)

            st.subheader(f"{platform} Posts")
            st.markdown(content)

        elif content_type == "Marketing Email":
            content = generate_email(topic, audience, tone, email_type, brand_voice)

            st.subheader("Generated Email")
            st.markdown(content)

    # Export
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download as Markdown",
            data=content,
            file_name=f"{content_type.lower().replace(' ', '_')}_{topic[:20].replace(' ', '_')}.md",
            mime="text/markdown",
        )
    with col2:
        word_ct = len(content.split())
        char_ct = len(content)
        st.caption(f"Words: {word_ct} | Characters: {char_ct}")

elif generate_btn:
    st.warning("Please fill in the topic and target audience.")
else:
    st.info("Configure your content settings in the sidebar and click **Generate Content** to start.")

    st.subheader("What You Can Generate")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Blog Posts**\n\nSEO-optimized articles with structured headings, keyword integration, and calls to action.")
    with col2:
        st.markdown("**Social Media**\n\nPlatform-specific posts for Twitter/X, LinkedIn, and Instagram with hashtags and engagement hooks.")
    with col3:
        st.markdown("**Marketing Emails**\n\nSubject lines, preview text, body copy, and CTA buttons for any campaign type.")
```

### How to Run

```bash
pip install streamlit langchain langchain-openai python-dotenv
export OPENAI_API_KEY="sk-..."
streamlit run content_generator.py
```

### How to Extend It

- Add a "content calendar" feature that plans a week or month of posts across channels.
- Integrate image generation (DALL-E or Stable Diffusion) for social media visuals.
- Add a brand voice training feature where the user pastes examples of existing content and the system extracts style guidelines.
- Store generated content in a database to build a searchable content library.
- Add plagiarism checking by comparing generated content against known sources.

---

# PART 2: Interview Preparation

---

## Top 50 GenAI Interview Questions & Answers

### Fundamentals (Questions 1-10)

**Q1: What is Generative AI?**
Generative AI refers to artificial intelligence systems that can create new content -- text, images, audio, code, or video -- rather than simply classifying or predicting from existing data. It learns patterns from training data and produces novel outputs that follow those patterns.

**Q2: What is a Large Language Model (LLM)?**
An LLM is a neural network trained on massive text corpora (billions to trillions of tokens) to understand and generate human language. Examples include GPT-4, Claude, Llama, and Gemini. They work by predicting the next token in a sequence, but this simple objective produces emergent capabilities like reasoning, translation, and code generation.

**Q3: What is the Transformer architecture?**
The Transformer is the neural network architecture behind modern LLMs, introduced in the 2017 paper "Attention Is All You Need." It replaces recurrent processing with self-attention mechanisms that process all tokens in parallel. Key components are multi-head self-attention, position embeddings, feed-forward layers, and layer normalization. It has an encoder-decoder structure, though many LLMs use decoder-only variants.

**Q4: How does the attention mechanism work?**
Attention computes a weighted sum of value vectors, where the weights are determined by the similarity between query and key vectors. For each token, Q, K, and V matrices are computed. The attention score is softmax(QK^T / sqrt(d_k)) * V. This allows each token to attend to every other token in the sequence, capturing long-range dependencies. Multi-head attention runs this in parallel across multiple representation subspaces.

**Q5: What is the difference between GPT and BERT?**
GPT (decoder-only) is autoregressive: it predicts the next token given all previous tokens, making it suited for generation. BERT (encoder-only) is bidirectional: it looks at tokens in both directions simultaneously, making it suited for understanding tasks like classification and NER. GPT generates text left-to-right; BERT fills in masked tokens using full context.

**Q6: What is pre-training vs. fine-tuning?**
Pre-training is the initial phase where a model learns general language understanding from a massive, diverse corpus using self-supervised objectives (next-token prediction). Fine-tuning is a subsequent phase where the pre-trained model is further trained on a smaller, task-specific dataset to specialize its behavior -- for example, training on instruction-following data or medical texts.

**Q7: What is RLHF?**
Reinforcement Learning from Human Feedback is a training technique used to align LLMs with human preferences. The process has three steps: (1) supervised fine-tuning on demonstration data, (2) training a reward model on human preference comparisons, and (3) optimizing the LLM against the reward model using PPO or similar RL algorithms. This is what makes models helpful, harmless, and honest.

**Q8: What is a token?**
A token is the basic unit of text that an LLM processes. Depending on the tokenizer, a token can be a word, part of a word, or a single character. Most modern LLMs use subword tokenization (BPE or SentencePiece). A rough rule of thumb is 1 token is approximately 4 characters or 0.75 English words. Tokenization matters because context windows and pricing are measured in tokens.

**Q9: What is a context window?**
The context window is the maximum number of tokens an LLM can process in a single call, including both the input prompt and the generated output. GPT-4o supports 128K tokens, Claude supports up to 200K tokens. Longer context windows allow processing larger documents but increase compute cost and can degrade attention quality on very long inputs.

**Q10: What is the difference between zero-shot, few-shot, and fine-tuning?**
Zero-shot means giving the model a task description with no examples. Few-shot means including a handful of examples in the prompt to guide the model. Fine-tuning means actually updating the model's weights on task-specific data. Zero-shot is cheapest but least reliable; fine-tuning is most expensive but most consistent. Few-shot is the practical middle ground for most use cases.

### Technical (Questions 11-25)

**Q11: Explain the RAG pipeline.**
Retrieval-Augmented Generation has three stages: (1) Indexing -- documents are chunked, embedded into vectors, and stored in a vector database. (2) Retrieval -- when a user asks a question, the query is embedded and the most similar document chunks are retrieved via vector similarity search. (3) Generation -- the retrieved chunks are injected into the LLM prompt as context, and the LLM generates an answer grounded in that context. RAG reduces hallucination and allows the model to access up-to-date or proprietary information.

**Q12: What are embeddings?**
Embeddings are dense vector representations of text (or images, audio, etc.) in a continuous high-dimensional space. Semantically similar texts have vectors that are close together (measured by cosine similarity). Models like `text-embedding-3-small` convert text to vectors of 1536 dimensions. They enable semantic search, clustering, and the retrieval step in RAG.

**Q13: What is a vector database?**
A vector database is a specialized data store optimized for storing, indexing, and querying high-dimensional vectors. Unlike traditional databases that use exact-match queries, vector databases use approximate nearest neighbor (ANN) search algorithms like HNSW or IVF. Examples include ChromaDB, Pinecone, Weaviate, Qdrant, and pgvector. They are the backbone of RAG systems.

**Q14: What chunking strategies exist and when do you use each?**
Fixed-size chunking splits text every N characters with overlap -- simple and works well for homogeneous documents. Recursive character splitting tries to split on natural boundaries (paragraphs, sentences) while staying near the target size -- best general-purpose strategy. Semantic chunking uses embeddings to group sentences that are semantically similar -- best for documents with varied topics. Document-specific chunking respects structure (Markdown headers, HTML tags, code blocks). The right choice depends on document structure and query patterns.

**Q15: What is temperature in LLM inference?**
Temperature controls the randomness of token selection. It scales the logits before the softmax function. Temperature 0 makes the model deterministic (always picks the highest-probability token). Temperature 1 is the default distribution. Higher temperatures (e.g., 1.5) increase randomness and creativity but can produce incoherent text. For factual tasks use low temperature (0-0.2); for creative tasks use higher (0.7-1.0).

**Q16: What is top-p (nucleus) sampling?**
Top-p sampling selects from the smallest set of tokens whose cumulative probability exceeds the threshold p. For example, top-p=0.9 means the model considers only tokens that together account for 90% of the probability mass, then samples from that set. It dynamically adjusts the candidate pool size -- using fewer candidates when the model is confident and more when it is uncertain. It is often preferred over top-k because of this adaptiveness.

**Q17: How does fine-tuning differ from prompt engineering?**
Prompt engineering modifies the input to guide the model's behavior without changing its weights. Fine-tuning updates the model's weights on new data. Prompt engineering is faster, cheaper, and requires no ML infrastructure. Fine-tuning is better when you need consistent behavior, domain-specific knowledge, or a particular output format across thousands of calls. Start with prompt engineering; fine-tune only when it is insufficient.

**Q18: What is LoRA and why does it matter?**
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method. Instead of updating all model weights (billions of parameters), it freezes the original weights and injects small trainable low-rank matrices into each layer. This reduces the number of trainable parameters by 10,000x, cuts GPU memory requirements dramatically, and makes fine-tuning feasible on consumer hardware. QLoRA adds quantization on top for even greater efficiency.

**Q19: What is a hallucination and how do you mitigate it?**
A hallucination is when an LLM generates plausible-sounding but factually incorrect information. Mitigation strategies include: RAG to ground responses in retrieved facts, lowering temperature, asking the model to cite sources, adding instructions like "if you don't know, say so," implementing fact-checking pipelines, and using structured outputs to constrain responses. No approach eliminates hallucinations entirely; defense in depth is required.

**Q20: Explain the difference between semantic search and keyword search.**
Keyword search (BM25, TF-IDF) matches exact or stemmed terms -- it fails when the query uses different words than the document. Semantic search encodes both the query and documents as embeddings and measures vector similarity -- it understands that "automobile" and "car" are related even without shared keywords. In practice, hybrid search (combining both) often outperforms either alone.

**Q21: What is an AI agent?**
An agent is an LLM that can take actions in a loop: observe (receive input), reason (decide what to do), act (call tools/APIs), and observe the result. Unlike a simple chain, an agent dynamically decides which tools to use and how many steps to take. Frameworks like LangChain's AgentExecutor, LangGraph, and CrewAI implement the agent pattern. Common agent architectures include ReAct (Reasoning + Acting) and Plan-and-Execute.

**Q22: What is function calling / tool use in LLMs?**
Function calling is a feature where the LLM outputs structured JSON representing a function invocation rather than plain text. The developer defines available functions with their schemas, the LLM decides when and how to call them, and the application executes the function and feeds the result back. This enables LLMs to interact with databases, APIs, calculators, and other systems reliably.

**Q23: How do you evaluate RAG system quality?**
Key metrics: (1) Retrieval quality -- recall@k (did the right chunks get retrieved?), precision@k, MRR. (2) Generation quality -- faithfulness (does the answer contradict the retrieved context?), answer relevancy, completeness. (3) End-to-end -- correctness vs. ground truth, user satisfaction. Tools like RAGAS and DeepEval automate these evaluations. Always test with a representative set of question-answer pairs.

**Q24: What is prompt injection and how do you defend against it?**
Prompt injection is an attack where a user crafts input that overrides the system prompt, causing the LLM to ignore instructions or reveal confidential information. Defenses include: separating system and user prompts clearly, input validation and sanitization, output filtering, using a secondary LLM to detect injection attempts, limiting the model's access to sensitive tools, and never putting secrets in system prompts.

**Q25: What is quantization in the context of LLMs?**
Quantization reduces model precision from 32-bit or 16-bit floating point to lower bit widths (8-bit, 4-bit, or even 2-bit). This shrinks model size, reduces memory usage, and speeds up inference with minimal quality loss. GPTQ, AWQ, and bitsandbytes are popular quantization methods. A 70B parameter model in 4-bit quantization fits in about 35GB of RAM instead of 140GB. This is what makes running large models on consumer GPUs possible.

### Application / Practical (Questions 26-40)

**Q26: What is LangChain and what problem does it solve?**
LangChain is a framework for building LLM-powered applications. It provides abstractions for prompts, chains (sequences of LLM calls), agents (LLMs that use tools), memory (conversation state), and retrieval (RAG). It solves the problem of gluing together LLM calls, vector stores, APIs, and tools into a coherent application with reusable components.

**Q27: Walk through building a RAG application step by step.**
(1) Collect and load documents using document loaders. (2) Split documents into chunks using a text splitter (typically 500-1000 chars with overlap). (3) Generate embeddings for each chunk. (4) Store embeddings in a vector database. (5) At query time, embed the user question. (6) Retrieve top-k similar chunks via vector search. (7) Construct a prompt with the retrieved context and user question. (8) Send to the LLM and return the answer. (9) Optionally, add conversation memory and source citations.

**Q28: How do you choose the right chunk size?**
Chunk size is a trade-off. Smaller chunks (200-500 chars) give more precise retrieval but may lose context. Larger chunks (1000-2000 chars) preserve more context but may include irrelevant information and reduce retrieval precision. Start with 500-1000 characters and 10-20% overlap. Test retrieval quality on representative queries and adjust. For code, respect function boundaries. For legal documents, respect paragraph or section boundaries.

**Q29: What is prompt engineering and what are key techniques?**
Prompt engineering is the practice of designing inputs to get optimal outputs from LLMs. Key techniques: (1) Be specific and explicit in instructions. (2) Use few-shot examples to demonstrate desired output. (3) Chain-of-thought prompting: ask the model to reason step by step. (4) Role prompting: assign a persona. (5) Output formatting: specify JSON, Markdown, etc. (6) Negative prompting: state what not to do. (7) Iterative refinement: test and adjust based on failure cases.

**Q30: How do you handle conversation memory in a chatbot?**
Options include: (1) Buffer memory -- store the full conversation (simple but hits context limits). (2) Window memory -- keep only the last N exchanges. (3) Summary memory -- use the LLM to summarize older conversation and keep only the summary. (4) Entity memory -- track key entities mentioned across the conversation. (5) Vector-backed memory -- embed past messages and retrieve relevant ones. LangChain provides implementations of all these patterns.

**Q31: How would you deploy an LLM application to production?**
Key considerations: (1) Serve via a REST API using FastAPI or Flask. (2) Use async/streaming for long responses. (3) Add rate limiting and authentication. (4) Cache frequent queries to reduce API costs. (5) Monitor latency, error rates, and token usage. (6) Log inputs and outputs for debugging (with PII filtering). (7) Set up fallback models in case the primary LLM is down. (8) Use Docker and container orchestration for scaling. (9) Implement guardrails for content safety.

**Q32: How do you optimize LLM API costs?**
Strategies: (1) Use smaller models when possible (GPT-4o-mini instead of GPT-4o). (2) Cache responses for repeated queries. (3) Reduce prompt length by removing unnecessary instructions. (4) Batch requests where possible. (5) Use streaming to fail fast on bad outputs. (6) Fine-tune a smaller model for high-volume tasks. (7) Set max_tokens to avoid overly long responses. (8) Use tiered approaches -- cheap model first, expensive model only when needed.

**Q33: What is the difference between chains and agents in LangChain?**
A chain is a fixed sequence of steps executed in order -- for example, retrieve context, format prompt, call LLM. The flow is deterministic. An agent is dynamic: the LLM decides at each step which tool to call, examines the result, and decides the next action. Chains are predictable and easier to debug. Agents are flexible and can solve open-ended problems but are harder to control and may loop or fail unpredictably.

**Q34: How do you implement streaming responses?**
Streaming sends tokens to the client as they are generated, rather than waiting for the complete response. With OpenAI's API, set `stream=True` and iterate over chunks. In LangChain, use `.stream()` on chains or models. On the frontend, use Server-Sent Events (SSE) or WebSockets. Streaming dramatically improves perceived latency -- the user starts reading immediately instead of waiting 5-10 seconds for a full response.

**Q35: What is a guardrail in GenAI applications?**
Guardrails are programmatic checks on LLM inputs and outputs to enforce safety, accuracy, and compliance. Input guardrails validate and sanitize user messages (profanity filters, PII detection, prompt injection detection). Output guardrails check the LLM's response (factual consistency, format compliance, content policy). Libraries like Guardrails AI and NeMo Guardrails provide frameworks for building these checks.

**Q36: How do you handle structured output from LLMs?**
Methods: (1) Instruct the model to respond in JSON and parse it. (2) Use OpenAI's function calling / response_format to enforce JSON schema. (3) Use Pydantic models with LangChain's structured output parsers. (4) Use constrained generation (grammar-based decoding) with local models. (5) Validate output with retry logic -- if parsing fails, re-prompt with the error message. Structured output is essential for any LLM that feeds into downstream code.

**Q37: What are the main LLM evaluation metrics?**
Automated metrics: BLEU, ROUGE, BERTScore (for text similarity to reference), perplexity (for language modeling quality). Human evaluation: fluency, relevance, factual accuracy, harmlessness. Task-specific: for RAG use faithfulness and answer relevancy; for summarization use coverage and conciseness; for code generation use pass@k. In practice, LLM-as-judge (using a strong model to evaluate a weaker one) is increasingly common for scalable evaluation.

**Q38: How do you build a multi-modal GenAI application?**
Multi-modal apps process and generate multiple data types (text + images, text + audio, etc.). GPT-4o and Claude accept both text and image inputs natively. For image generation, integrate DALL-E or Stable Diffusion APIs. For audio, use Whisper for speech-to-text and TTS APIs for text-to-speech. The application orchestrates these models -- for example, accepting a voice query, transcribing it, processing it with a text LLM, and generating both a text and image response.

**Q39: What is LangGraph and when would you use it?**
LangGraph is a library for building stateful, multi-step LLM applications as graphs. Each node is a function or LLM call, and edges define the flow. Unlike chains (linear) or agents (LLM-controlled), LangGraph gives the developer explicit control over branching, looping, and state management. Use it when you need complex workflows like: human-in-the-loop approval steps, parallel tool execution, conditional branching, or cyclical agent loops with explicit termination conditions.

**Q40: How do you handle PII in LLM applications?**
Strategies: (1) Detect and redact PII before sending to the LLM (using libraries like Presidio or custom regex). (2) Use data masking -- replace real names with placeholders, process, then reverse the mapping. (3) Use local or on-premise models for sensitive data instead of cloud APIs. (4) Implement data retention policies -- don't log prompts containing PII. (5) Use API options like OpenAI's data usage policies (opt out of training). (6) Encrypt data at rest and in transit.

### System Design (Questions 41-50)

**Q41: Design a customer support chatbot for an e-commerce company.**
Components: (1) Knowledge base -- product catalog, FAQ, policies stored in a vector DB, updated nightly. (2) Intent classifier -- determine if the query is about orders, products, returns, or general. (3) RAG pipeline -- retrieve relevant docs and generate answer. (4) Order lookup integration -- API call to check order status using user's order ID. (5) Escalation logic -- if sentiment is negative or after 3 failed attempts, route to human. (6) Conversation memory -- maintain session context. (7) Analytics -- log queries, track resolution rate, identify gaps in the knowledge base. Trade-offs: latency vs. accuracy (use a fast model for simple queries, powerful model for complex ones).

**Q42: Design a RAG system for a company with 10 million documents.**
Key considerations: (1) Use a scalable vector database like Pinecone or Weaviate, not SQLite-backed ChromaDB. (2) Implement hierarchical indexing -- first retrieve relevant document clusters, then search within them. (3) Use metadata filtering to narrow search space (department, date range, document type). (4) Implement hybrid search (vector + keyword) for better recall. (5) Cache frequently asked queries. (6) Use asynchronous indexing -- new documents are queued and embedded in the background. (7) Implement access control so users only see documents they are authorized to view. (8) Monitor and retrain embeddings as content evolves.

**Q43: Design a content moderation pipeline for a social media platform.**
Architecture: (1) First pass -- fast regex and keyword filter catches obvious violations (near-zero latency). (2) Second pass -- lightweight classifier (fine-tuned BERT) categorizes content into risk levels (low latency). (3) Third pass -- for medium-risk content, use an LLM to evaluate context and nuance (higher latency, reserved for ambiguous cases). (4) Human review queue for edge cases flagged by the LLM. (5) Feedback loop: human decisions feed back into classifier training. Trade-offs: false positive rate (over-censoring) vs. false negative rate (missing violations). Use tiered approach to balance cost with thoroughness.

**Q44: Design an AI-powered code review system.**
Components: (1) Git webhook triggers review on PR creation. (2) Diff parser extracts changed files and functions. (3) Context assembler retrieves related code (imports, callers, tests) for full picture. (4) LLM reviews diff with context for bugs, style issues, security vulnerabilities, and performance concerns. (5) Comment formatter posts inline comments on specific lines via GitHub API. (6) Severity classifier categorizes findings as critical, warning, or suggestion. Trade-offs: avoid being too noisy (developers ignore tools that flag too many false positives). Start with high-precision rules and expand gradually.

**Q45: Design a real-time translation system using LLMs.**
Architecture: (1) Input detection -- identify source language using a fast classifier or the LLM itself. (2) Streaming translation -- use streaming API to translate in real-time as the user types or speaks. (3) Terminology database -- company-specific terms are stored in a lookup table and injected into the prompt to ensure consistent translations. (4) Post-editing interface -- bilingual reviewers can correct translations, and corrections are stored for few-shot examples. (5) Caching layer -- cache translations of common phrases. Trade-offs: LLM translation is high quality but expensive; for high-volume use cases, use a dedicated translation model (NLLB, MarianMT) and reserve the LLM for quality-critical content.

**Q46: Design a document summarization service that handles 1000 documents per hour.**
Architecture: (1) Queue-based processing -- documents enter a message queue (SQS, RabbitMQ). (2) Worker pool -- multiple workers consume from the queue and process in parallel. (3) For short documents (under context window), summarize in a single LLM call. (4) For long documents, use map-reduce: split into sections, summarize each section, then summarize the summaries. (5) Use a smaller, faster model (GPT-4o-mini) for initial summaries and a stronger model for the final merge. (6) Cache summaries keyed by document hash. (7) Expose via async API -- return a job ID immediately, client polls for completion.

**Q47: Design a personalized learning assistant.**
Components: (1) Student profile -- tracks knowledge level, learning style, and progress. (2) Curriculum graph -- topics organized as a prerequisite graph. (3) Adaptive questioning -- LLM generates questions at the appropriate difficulty level based on the student's history. (4) Explanation engine -- when the student answers wrong, the LLM explains using the student's preferred style (visual, step-by-step, analogy-based). (5) Spaced repetition scheduler -- resurfaces topics at optimal intervals. (6) RAG over textbook content for grounded explanations. Trade-offs: over-adaptation (student never challenged) vs. under-adaptation (student always frustrated).

**Q48: Design an AI pipeline that extracts structured data from invoices.**
Architecture: (1) OCR layer -- convert PDF/image invoices to text using Tesseract or cloud OCR (AWS Textract). (2) LLM extraction -- send OCR text to an LLM with a structured output schema (vendor name, date, line items, amounts, tax, total). (3) Validation layer -- check that line items sum to total, dates are valid, vendor exists in master data. (4) Confidence scoring -- flag low-confidence extractions for human review. (5) Feedback loop -- corrected extractions become few-shot examples. Use function calling / structured output to ensure consistent JSON output. Trade-off: accuracy vs. cost -- fine-tune a small model for high-volume scenarios.

**Q49: Design a multi-agent system for automated research.**
Architecture: (1) Orchestrator agent -- receives the research question and creates a plan. (2) Search agent -- queries multiple sources (web search, academic papers, internal docs). (3) Analysis agent -- reads retrieved documents and extracts key findings. (4) Synthesis agent -- combines findings into a coherent report with citations. (5) Critic agent -- reviews the report for factual consistency and gaps. (6) Communication via shared state (LangGraph) or message passing. Each agent has specialized tools and prompts. Trade-offs: more agents add latency and cost; fewer agents reduce quality. Use parallel execution where possible.

**Q50: How would you migrate a traditional search system to semantic search?**
Phased approach: (1) Phase 1 -- add semantic search alongside existing keyword search as a secondary ranking signal. Run A/B tests. (2) Phase 2 -- implement hybrid search (combine BM25 and vector scores using reciprocal rank fusion). (3) Phase 3 -- add query understanding (use LLM to expand or rephrase queries before search). (4) Phase 4 -- add generative answers on top of search results (RAG). Key infrastructure: embed all existing documents (batch job), set up incremental indexing for new documents, choose a vector database that can scale to your corpus size. Monitor click-through rates and query-to-result relevance at each phase.

---

## How to Explain Your Projects in Interviews

Interviewers evaluate not just what you built, but how you think. Use this adapted STAR method for technical projects.

### The STAR-T Framework for Technical Projects

**Situation**: What was the problem or need? Who was the intended user?

> "We needed a way for our legal team to query across hundreds of contract PDFs without reading each one manually."

**Task**: What was your specific role and goal?

> "I designed and built a multi-document research assistant that lets users upload PDFs and ask natural language questions with source citations."

**Action**: What technical decisions did you make and why?

> "I used LangChain for orchestration and ChromaDB as the vector store because the dataset was under 100K documents and we needed a local solution. I chose recursive character splitting at 1000 characters with 200-character overlap after testing showed it gave better retrieval precision than 500-character chunks for our legal documents. I implemented conversation memory using a sliding window of 8 exchanges to stay within the context limit."

**Result**: What was the outcome? Use numbers when possible.

> "The tool reduced document review time by approximately 60%. The team went from spending 2 hours searching for specific clauses to getting answers in under 10 seconds."

**Trade-offs**: What alternatives did you consider and why did you reject them?

> "I considered Pinecone instead of ChromaDB for scalability, but our data was sensitive and needed to stay on-premises. I considered fine-tuning instead of RAG, but our documents change monthly and re-training would be impractical."

### What Interviewers Look For

1. **Understanding of trade-offs** -- not just what you built, but why you chose one approach over another.
2. **Awareness of limitations** -- can you identify where your system would fail?
3. **Production thinking** -- did you consider error handling, cost, latency, security, and scalability?
4. **Depth of understanding** -- can you explain what happens inside the vector database or how attention works, not just how to call the API?
5. **Iteration mindset** -- did you test different approaches and measure results?

### Common Follow-Up Questions and How to Handle Them

- "What would you change if you had to handle 10x the data?" -- Discuss scaling the vector DB, adding metadata filtering, using async processing.
- "How do you handle hallucinations?" -- Describe your grounding strategy (RAG, citations, confidence thresholds, human-in-the-loop).
- "What happens when the LLM API goes down?" -- Discuss fallback models, caching, graceful degradation.
- "How would you test this?" -- Describe your evaluation dataset, retrieval metrics (recall@k), and generation quality metrics (faithfulness).

---

## Key Concepts Cheat Sheet

| Term | One-Liner Definition |
|---|---|
| **Transformer** | Neural network architecture using self-attention to process sequences in parallel; foundation of all modern LLMs. |
| **Self-Attention** | Mechanism that lets each token weigh the importance of every other token in the sequence. |
| **Multi-Head Attention** | Running multiple attention operations in parallel to capture different types of relationships. |
| **Embedding** | Dense vector representation of text (or other data) where semantic similarity corresponds to vector proximity. |
| **Vector Database** | Database optimized for storing and querying high-dimensional vectors using approximate nearest neighbor search. |
| **RAG** | Retrieval-Augmented Generation: retrieving relevant documents and including them in the LLM prompt to ground responses in facts. |
| **Fine-Tuning** | Further training a pre-trained model on task-specific data to specialize its behavior. |
| **LoRA** | Low-Rank Adaptation: parameter-efficient fine-tuning that trains small adapter matrices instead of all model weights. |
| **QLoRA** | LoRA combined with 4-bit quantization, enabling fine-tuning of large models on consumer GPUs. |
| **Agent** | An LLM that can reason, plan, and take actions by calling external tools in a loop until a task is complete. |
| **Chain** | A fixed sequence of LLM calls and processing steps executed in a deterministic order. |
| **Token** | The basic unit of text processed by an LLM; roughly 4 characters or 0.75 words in English. |
| **Context Window** | Maximum number of tokens an LLM can process in a single request (input + output combined). |
| **Hallucination** | When an LLM generates plausible-sounding but factually incorrect information. |
| **Grounding** | Techniques to anchor LLM outputs in verified facts, such as RAG or tool use. |
| **RLHF** | Reinforcement Learning from Human Feedback: aligning LLM behavior with human preferences using a reward model. |
| **Temperature** | Parameter controlling randomness in token selection; 0 = deterministic, higher = more random. |
| **Top-p Sampling** | Nucleus sampling: selecting from the smallest token set whose cumulative probability exceeds p. |
| **Top-k Sampling** | Selecting from only the k most probable next tokens. |
| **Prompt Engineering** | Designing LLM inputs to elicit optimal outputs without changing model weights. |
| **Few-Shot Learning** | Including examples in the prompt to guide the model's output format and behavior. |
| **Chain-of-Thought** | Prompting technique where the model is asked to reason step by step before giving a final answer. |
| **Tokenizer** | Algorithm that converts raw text into tokens; common approaches are BPE and SentencePiece. |
| **Chunking** | Splitting documents into smaller segments for embedding and retrieval in RAG pipelines. |
| **Cosine Similarity** | Metric measuring the angle between two vectors; used to rank search results in semantic search. |
| **ANN Search** | Approximate Nearest Neighbor: efficient algorithm for finding similar vectors in large collections (e.g., HNSW, IVF). |
| **Quantization** | Reducing model weight precision (32-bit to 8-bit or 4-bit) to shrink model size and speed up inference. |
| **Prompt Injection** | Attack where user input manipulates the LLM into ignoring system instructions. |
| **Guardrails** | Programmatic checks on LLM inputs and outputs to enforce safety, accuracy, and format compliance. |
| **Structured Output** | Constraining LLM responses to a defined format (JSON, XML) for reliable downstream processing. |
| **Function Calling** | LLM capability to output structured tool invocations instead of plain text, enabling interaction with external systems. |
| **Streaming** | Sending LLM output tokens to the client incrementally as they are generated rather than waiting for completion. |
| **LangChain** | Framework providing abstractions for building LLM applications with chains, agents, memory, and retrieval. |
| **LangGraph** | Library for building stateful, graph-based LLM workflows with explicit control over branching and looping. |
| **Semantic Search** | Finding documents by meaning rather than keyword matching, using embedding similarity. |
| **Hybrid Search** | Combining keyword search (BM25) and semantic search (vectors) for better retrieval accuracy. |
| **Reward Model** | A model trained on human preference data to score LLM outputs, used in the RLHF pipeline. |
| **Instruction Tuning** | Fine-tuning a base model on instruction-response pairs to make it follow user instructions reliably. |
| **DPO** | Direct Preference Optimization: a simpler alternative to RLHF that skips the reward model and optimizes preferences directly. |

---

## Conclusion

You now have three portfolio-ready projects that demonstrate the most in-demand GenAI skills: RAG, vector databases, conversational AI, sentiment analysis, prompt engineering, and user-facing application design. You also have a bank of 50 interview questions covering fundamentals through system design, a framework for discussing your projects with confidence, and a cheat sheet for quick review.

Build these projects, deploy them, put them on your resume, and be ready to explain every technical decision you made. That combination of hands-on work and articulate communication is what separates candidates who get offers from those who do not.
