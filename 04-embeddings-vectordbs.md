# Chapter 4: Embeddings & Vector Databases

---

## 1. Concept Explanation

### What Are Embeddings?

At its core, an **embedding** is a way to turn words, sentences, or entire documents into lists of numbers (vectors) that capture their *meaning*.

Why do we need this? Computers do not understand language the way we do. To a computer, "happy" and "joyful" are just two unrelated strings of characters. Embeddings solve this problem by mapping text into a numerical space where **meaning is encoded as position**.

**Analogy: GPS Coordinates for Meaning**

Think of embeddings like GPS coordinates, but instead of mapping physical locations, they map *meanings*. In this "meaning space":

- "king" and "queen" have coordinates that are close together (both are royalty).
- "king" and "banana" have coordinates that are far apart (royalty vs. fruit).
- "dog" and "puppy" sit near each other, while "dog" and "spreadsheet" are in entirely different neighborhoods.

The beautiful thing is that relationships are preserved. The classic example: if you take the vector for "king," subtract "man," and add "woman," you land near "queen." Arithmetic on meaning!

### What Are Vector Databases?

A **vector database** is a specialized database designed to store, index, and search these embedding vectors efficiently. Traditional databases are great at exact lookups ("find the user with id=42"), but they cannot answer questions like "find me products *similar to* this description." Vector databases can, because they search by *meaning proximity* rather than exact matching.

---

## 2. Detailed Notes

### Word Embeddings vs. Sentence Embeddings

**Word Embeddings** represent individual words as vectors:

- **Word2Vec** (Google, 2013) — learns word vectors by predicting surrounding words (Skip-gram) or predicting a word from its context (CBOW).
- **GloVe** (Stanford) — learns from global word co-occurrence statistics across a corpus.

**Sentence/Document Embeddings** represent entire chunks of text as a single vector:

- **Sentence-Transformers** — fine-tuned transformer models that produce a single vector for a full sentence or paragraph.
- **OpenAI Embeddings** — API-based models like `text-embedding-ada-002` that return a vector for any input text.

Word embeddings were groundbreaking, but modern applications (semantic search, RAG) overwhelmingly use sentence-level embeddings because meaning often depends on the full context, not individual words.

### How Embeddings Are Created

Embedding models are **trained on massive text datasets**. During training, the model learns to place texts with similar meanings near each other in vector space and texts with different meanings far apart. The model is never told "these two sentences are similar" explicitly — it learns patterns from the structure of language itself (self-supervised learning).

### Dimensionality

Each embedding is a list of floating-point numbers. The length of that list is the **dimensionality**:

| Model | Dimensions |
|---|---|
| `all-MiniLM-L6-v2` (sentence-transformers) | 384 |
| `all-mpnet-base-v2` (sentence-transformers) | 768 |
| `text-embedding-ada-002` (OpenAI) | 1536 |
| `text-embedding-3-small` (OpenAI) | 1536 |

Higher dimensions can capture more nuance but require more storage and compute.

### Similarity Metrics

Once you have two vectors, how do you measure how "close" they are?

- **Cosine Similarity** — measures the angle between two vectors. Returns a value between -1 and 1 (1 = identical direction, 0 = unrelated, -1 = opposite). This is the most commonly used metric because it is not affected by vector magnitude, only direction.
- **Dot Product** — sum of element-wise products. Similar to cosine similarity but affected by magnitude. Works well when vectors are already normalized.
- **Euclidean Distance** — straight-line distance between two points. Smaller = more similar. Less common for text embeddings because it is sensitive to vector magnitude.

### Vector Databases

| Database | Type | Key Feature |
|---|---|---|
| **ChromaDB** | Open-source, local | Beginner-friendly, Python-native |
| **FAISS** | Library (Meta) | Blazing fast, runs in-memory |
| **Pinecone** | Managed cloud service | Fully hosted, scales automatically |
| **Weaviate** | Open-source, self-hosted or cloud | Supports hybrid search (vector + keyword) |

### How Vector Search Works

Searching through millions of vectors by computing similarity to every single one would be extremely slow. Vector databases use **Approximate Nearest Neighbors (ANN)** algorithms to speed this up. Techniques like HNSW (Hierarchical Navigable Small World) build graph structures that let you jump quickly to the right neighborhood of vectors, trading a tiny bit of accuracy for massive speed gains.

### Popular Embedding Models

- **OpenAI `text-embedding-ada-002`** — general-purpose, 1536 dimensions, easy API access.
- **Sentence-Transformers (`all-MiniLM-L6-v2`)** — free, open-source, runs locally, 384 dimensions, fast.
- **Cohere Embed** — API-based, good multilingual support.

---

## 3. Visual/Intuitive Explanation

### The Point Cloud Mental Model

Imagine a 3D scatter plot (even though real embeddings have hundreds of dimensions, the idea is the same). Each dot is a word or sentence:

```
        [animals cluster]
           dog *
         cat *    * puppy
       horse *

  [food cluster]              [technology cluster]
    pizza *                      laptop *
   burger *  * sushi           python *  * java
     rice *                    server *
```

Words with related meanings form visible clusters. "Dog" is near "cat" and "puppy" but far from "pizza" and "laptop."

### How Semantic Search Works (Step by Step)

```
User query: "How do I feed my pet?"
         |
         v
  [1] Embed the query → [0.12, -0.34, 0.87, ...]
         |
         v
  [2] Search the vector database for nearest vectors
         |
         v
  [3] Return the top-k most similar stored items:
       - "Best dog food brands for puppies" (similarity: 0.91)
       - "Cat feeding schedule guide"       (similarity: 0.88)
       - "Pet nutrition basics"             (similarity: 0.85)
```

### Why Cosine Similarity Works

Think of each embedding as an arrow pointing from the origin. Cosine similarity measures the **angle** between two arrows. If they point in the same direction (angle near 0), the texts mean similar things. If they point in perpendicular directions (angle near 90 degrees), they are unrelated. The length of the arrow does not matter — only the direction — which makes it robust for comparing texts of different lengths.

---

## 4. YouTube Resources

Search for these topics on YouTube:

- **"Word2Vec explained simply"** — look for videos that use the skip-gram/CBOW diagrams and the king-queen analogy.
- **"Vector databases explained for beginners"** — Fireship and similar channels have concise explainers.
- **"Embeddings crash course"** — search for "what are embeddings machine learning" for longer tutorials.
- **"Cosine similarity explained visually"** — helps solidify the geometric intuition.
- **"ChromaDB tutorial Python"** — hands-on walkthroughs for building your first vector search.

---

## 5. Official Documentation

| Resource | URL | What to Read |
|---|---|---|
| **ChromaDB Docs** | https://docs.trychroma.com | Start with the "Getting Started" guide. Learn `Collection`, `add`, and `query`. |
| **OpenAI Embeddings Guide** | https://platform.openai.com/docs/guides/embeddings | Understand model options, token limits, and best practices. |
| **Sentence-Transformers Docs** | https://www.sbert.net | Check the "Usage" section and the pre-trained models list. |
| **FAISS GitHub** | https://github.com/facebookresearch/faiss | Read the wiki for installation and the "Getting Started" tutorial. |

---

## 6. Code Examples

### Setup

```bash
pip install openai sentence-transformers chromadb numpy
```

### Generate Embeddings with Sentence-Transformers (Free, Local)

```python
from sentence_transformers import SentenceTransformer

# Load a lightweight model (runs locally, no API key needed)
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",
    "A kitten rested on the rug.",
    "Stock prices rose sharply today.",
]

# Generate embeddings — each is a 384-dimensional vector
embeddings = model.encode(sentences)

print(f"Shape: {embeddings.shape}")       # (3, 384)
print(f"First 5 values: {embeddings[0][:5]}")
```

### Generate Embeddings with OpenAI (API)

```python
from openai import OpenAI

client = OpenAI()  # expects OPENAI_API_KEY env variable

response = client.embeddings.create(
    input="The cat sat on the mat.",
    model="text-embedding-ada-002"
)

embedding = response.data[0].embedding
print(f"Dimensions: {len(embedding)}")  # 1536
```

### Calculate Cosine Similarity Manually

```python
import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    return dot_product / (magnitude_a * magnitude_b)

# Using the sentence-transformers embeddings from above
sim_cat_kitten = cosine_similarity(embeddings[0], embeddings[1])
sim_cat_stocks = cosine_similarity(embeddings[0], embeddings[2])

print(f"Cat vs Kitten:  {sim_cat_kitten:.4f}")  # ~0.75 (high — similar meaning)
print(f"Cat vs Stocks:  {sim_cat_stocks:.4f}")   # ~0.05 (low — unrelated)
```

### Store and Query Vectors with ChromaDB

```python
import chromadb

# Create an in-memory ChromaDB client
client = chromadb.Client()

# Create a collection (like a table)
collection = client.create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)

# Add documents — ChromaDB can auto-embed with its default model,
# but here we supply our own embeddings for clarity.
documents = [
    "Python is a great programming language.",
    "JavaScript is used for web development.",
    "Cats are wonderful pets.",
    "Dogs are loyal companions.",
    "Machine learning uses data to make predictions.",
]

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = model.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=doc_embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# Query: find documents similar to a new sentence
query = "What programming language should I learn?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

print("Top 3 results:")
for doc, dist in zip(results["documents"][0], results["distances"][0]):
    print(f"  [{dist:.4f}] {doc}")
```

### Build a Simple Semantic Search

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Our "database" of sentences
corpus = [
    "How to train a neural network from scratch",
    "Best Italian pasta recipes for beginners",
    "Introduction to quantum computing",
    "Tips for growing tomatoes in your garden",
    "Understanding backpropagation in deep learning",
    "How to make homemade pizza dough",
]

# Pre-compute embeddings for the entire corpus
corpus_embeddings = model.encode(corpus)

def search(query, top_k=3):
    query_embedding = model.encode([query])
    # Compute cosine similarity against all corpus embeddings
    similarities = np.dot(corpus_embeddings, query_embedding.T).squeeze()
    # Get indices of top-k results
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print(f"\nQuery: '{query}'")
    for idx in top_indices:
        print(f"  [{similarities[idx]:.4f}] {corpus[idx]}")

search("machine learning tutorial")
# Returns neural network and backpropagation results

search("cooking at home")
# Returns pasta, pizza, and tomato results
```

---

## 7. Mini Practice Tasks

**Task 1: Explore Similarity**
Take 10 sentences from different topics (sports, cooking, technology, travel, science). Embed them all, compute a full similarity matrix, and print a heatmap (use `matplotlib` or just print the numbers). Which pairs are most/least similar? Does it match your intuition?

**Task 2: Embedding Dimensions Comparison**
Embed the same 5 sentences using `all-MiniLM-L6-v2` (384-d) and `all-mpnet-base-v2` (768-d). Compare the cosine similarity scores. Does the higher-dimensional model produce noticeably different rankings?

**Task 3: Build a FAQ Bot**
Create a list of 10 FAQ question-answer pairs. When a user types a question, embed it, find the most similar FAQ question using cosine similarity, and return the corresponding answer. Test with paraphrased versions of the original questions.

**Task 4: ChromaDB with Metadata Filtering**
Store 15 documents in ChromaDB with metadata (e.g., `{"category": "tech"}`, `{"category": "food"}`). Query with both a semantic query and a metadata filter (e.g., "find similar documents but only in the tech category"). See how results change with and without the filter.

---

## 8. Quick Revision Summary

- **Embeddings** convert text into numerical vectors that encode meaning. Similar texts produce similar vectors.
- **Word embeddings** (Word2Vec, GloVe) handle individual words. **Sentence embeddings** (sentence-transformers, OpenAI) handle full sentences or paragraphs — these are more practical for modern applications.
- **Cosine similarity** is the go-to metric: it measures the angle between two vectors (1 = identical meaning, 0 = unrelated).
- **Vector databases** (ChromaDB, FAISS, Pinecone, Weaviate) store embeddings and enable fast similarity search using Approximate Nearest Neighbor algorithms.
- The typical workflow is: **text in → embedding model → vector → store in vector DB → query by similarity**.
- Embedding dimensions vary by model (384, 768, 1536). You must use the **same model** for both storing and querying.
- ANN algorithms trade a small amount of accuracy for massive speed gains when searching large collections.

---

## 9. Common Mistakes

**1. Mixing embedding models between indexing and querying.**
If you embed your documents with `all-MiniLM-L6-v2` (384-d) but embed your query with `text-embedding-ada-002` (1536-d), the vectors live in completely different spaces. Similarity scores will be meaningless. Always use the **same model** for everything in a given collection.

**2. Forgetting to normalize vectors before using dot product.**
Cosine similarity is self-normalizing (it divides by magnitudes). But if you use raw dot product for speed, your vectors must be unit-normalized first. Un-normalized vectors will bias results toward longer vectors, not more similar ones.

**3. Exceeding the model's token limit.**
Embedding models have maximum input lengths (e.g., 512 tokens for many sentence-transformers, 8191 tokens for OpenAI ada-002). Text beyond the limit is silently truncated, meaning you lose information. For long documents, split them into chunks first.

**4. Not chunking documents properly.**
Embedding an entire 10-page document into one vector dilutes the meaning. The embedding becomes a vague average of everything. Instead, split documents into meaningful chunks (paragraphs, sections, or sliding windows of 200-500 tokens) and embed each chunk separately.

**5. Using the wrong similarity metric for your database.**
If your vector database is configured for Euclidean distance but your application logic assumes cosine similarity, rankings will differ. Check your database configuration and be consistent.

**6. Storing raw text without metadata.**
When you retrieve a similar vector from the database, you often need more than just the text — you need the source document, page number, timestamp, or category. Always store useful metadata alongside your embeddings.

**7. Assuming embeddings understand everything.**
Embeddings capture semantic similarity, not factual correctness. The vectors for "The Earth orbits the Sun" and "The Sun orbits the Earth" will be very similar because the words are nearly identical. Embeddings do not reason about truth — they measure textual closeness.

---

*Next chapter: Chapter 5 — Retrieval-Augmented Generation (RAG), where we combine embeddings, vector databases, and LLMs to build systems that answer questions using your own data.*
