# Chapter 2: Introduction to Large Language Models (LLMs)

---

## 1. Concept Explanation

### What Is a Language Model?

A **language model** is a system trained to understand and generate human language. At its core, it learns the statistical patterns of text: given a sequence of words, it predicts what comes next. Think of it like **autocomplete on your phone, but 1000x smarter**. Your phone might suggest the next word in a text message. An LLM can write entire essays, answer complex questions, generate code, and carry on multi-turn conversations.

### What Makes It "Large"?

The "Large" in LLM refers to two things:

- **Training data**: LLMs are trained on massive datasets -- books, websites, code repositories, academic papers -- sometimes trillions of words.
- **Parameters**: The model itself contains billions (sometimes hundreds of billions) of numerical weights that encode patterns learned during training. GPT-3 has 175 billion parameters. More recent models are even larger.

These two factors combined allow LLMs to develop remarkably sophisticated language abilities that smaller models cannot achieve.

### How Do LLMs Generate Text?

LLMs work through **next-token prediction**. Given an input like "The capital of France is", the model calculates a probability distribution over its entire vocabulary and picks the most likely next token (e.g., "Paris"). It then appends that token to the input and repeats the process, generating text one token at a time.

This is an autoregressive process: each new token depends on everything that came before it.

### Pre-Training vs Fine-Tuning

LLMs are built in stages:

- **Pre-training**: The model reads vast amounts of text and learns general language patterns. This is extremely expensive (millions of dollars in compute). The result is a "base model" that can predict text but is not yet useful as an assistant.
- **Fine-tuning**: The pre-trained model is further trained on curated datasets -- often human-written examples of helpful, safe conversations. Techniques like RLHF (Reinforcement Learning from Human Feedback) teach the model to follow instructions and refuse harmful requests.

### Key Models to Know

| Model | Organization | Access |
|-------|-------------|--------|
| GPT-3.5 / GPT-4 / GPT-4o | OpenAI | API (closed-source) |
| Claude (Haiku, Sonnet, Opus) | Anthropic | API (closed-source) |
| LLaMA 2 / LLaMA 3 | Meta | Open-weight |
| Gemini (Nano, Pro, Ultra) | Google DeepMind | API (closed-source) |
| Mistral / Mixtral | Mistral AI | Open-weight |

---

## 2. Detailed Notes

### Tokenization

LLMs do not process raw text. They first break text into **tokens** -- small chunks that might be words, parts of words, or individual characters.

- **Byte-Pair Encoding (BPE)**: Used by GPT models. Starts with individual characters and iteratively merges the most frequent pairs. The word "unhappiness" might become `["un", "happiness"]` or `["un", "happ", "iness"]` depending on the vocabulary.
- **WordPiece**: Used by BERT and some Google models. Similar to BPE but uses a slightly different merging strategy based on likelihood rather than frequency.

Common English words usually map to a single token. Rare or technical words get split into multiple tokens. A rough rule of thumb: **1 token is about 0.75 words** (or 4 characters) in English.

### Context Windows

Every LLM has a **context window** -- the maximum number of tokens it can process in a single request (input + output combined). Examples:

- GPT-3.5: 4,096 or 16,385 tokens
- GPT-4o: 128,000 tokens
- Claude Sonnet/Opus: 200,000 tokens
- Gemini 1.5 Pro: 1,000,000+ tokens

If your input exceeds the context window, the model cannot process it. You must truncate or summarize.

### Temperature and Sampling

**Temperature** controls the randomness of the model's output:

- `temperature=0`: The model always picks the most probable token. Output is deterministic and repetitive.
- `temperature=0.7`: A balanced middle ground. Some creativity, mostly coherent.
- `temperature=1.0+`: Highly creative but potentially incoherent output.

Other sampling parameters include `top_p` (nucleus sampling) and `top_k`, which limit the pool of candidate tokens.

### Message Roles

Modern chat-based LLMs use a structured message format with three roles:

- **System**: Sets the behavior, persona, or rules for the assistant. Example: "You are a helpful Python tutor."
- **User**: The human's input or question.
- **Assistant**: The model's response.

### Chat vs Completion Models

- **Completion models** (legacy): Take raw text and continue it. No concept of roles.
- **Chat models** (modern standard): Take a structured list of messages with roles. Almost all current models use this format.

### Model Sizes and Parameters

A "parameter" is a single numerical weight in the neural network. More parameters generally means more capability but also more compute cost:

- 7B parameters: Can run on a good consumer GPU
- 13B-70B parameters: Requires specialized hardware or cloud GPUs
- 175B+ parameters: Only practical via API access

### Open-Source vs Closed-Source

- **Closed-source** (GPT-4, Claude, Gemini): You access them only through APIs. The weights are not available. The provider handles hosting and scaling.
- **Open-weight** (LLaMA, Mistral): Model weights are publicly available. You can download, modify, and run them locally. However, you need your own hardware.

### API-Based vs Local Models

- **API-based**: Send requests over the internet, pay per token. No hardware needed. Examples: OpenAI API, Anthropic API.
- **Local**: Download the model and run it on your own machine. Free after setup, but requires significant GPU memory. Tools like `ollama` and `llama.cpp` make this easier.

---

## 3. Visual/Intuitive Explanation

### Tokenization: From Sentence to Numbers

Imagine the sentence: **"LLMs are amazing!"**

```
Step 1 (Raw text):     "LLMs are amazing!"
Step 2 (Tokenize):     ["LL", "Ms", " are", " amazing", "!"]
Step 3 (Token IDs):    [4726, 4211, 527, 11914, 0]
```

The model never sees the original text. It operates entirely on these numerical IDs, processes them through billions of calculations, and outputs new numerical IDs that get decoded back into text.

### The Prediction Pipeline

```
Input tokens  -->  [Neural Network: billions of parameters]  -->  Probability distribution
                                                                   "Paris"  : 92%
                                                                   "Lyon"   : 3%
                                                                   "Berlin" : 1%
                                                                   ...
```

The model processes all input tokens simultaneously through layers of attention and feedforward networks, then produces a probability for every possible next token.

### Temperature: The Dice Analogy

Imagine you have a weighted die where one face ("Paris") has a 92% chance of landing face-up.

- **Temperature = 0**: You do not roll at all. You just pick "Paris" every time.
- **Temperature = 0.7**: You roll the die, but it is still heavily weighted. "Paris" comes up most of the time, but occasionally you get a surprise like "Lyon."
- **Temperature = 1.5**: You roll a nearly fair die. Any face could come up. The output becomes unpredictable and creative -- but also potentially wrong.

---

## 4. YouTube Resources

Search YouTube for these terms to find excellent explanations:

- **"Andrej Karpathy intro to large language models"** -- A world-class 1-hour overview from a former OpenAI researcher. Covers how LLMs work, their capabilities, and their limitations.
- **"Andrej Karpathy let's build GPT from scratch"** -- A hands-on coding walkthrough that builds a small GPT model in Python. Best watched after you understand the basics.
- **"What are large language models explained simply"** -- Multiple creators have beginner-friendly 10-15 minute explainers. Look for videos with high view counts and recent upload dates.
- **"How ChatGPT works technically"** -- Good for understanding the transformer architecture at an intuitive level.
- **"Tokenization explained NLP"** -- Helpful for understanding BPE and WordPiece in more depth.

---

## 5. Official Documentation

### OpenAI

- **Chat Completions API guide**: Read the "Chat Completions" section under the API reference. It explains message roles, temperature, token limits, and response formats. This is the single most important page for using GPT models programmatically.

### Anthropic

- **Messages API documentation**: Read the "Messages" section. Pay attention to how system prompts are handled (they are a separate parameter, not a message role). Also read the "Models" page to understand the differences between Claude Haiku, Sonnet, and Opus.

### Hugging Face

- **Model Hub** (huggingface.co/models): Browse open-weight models. Filter by task ("Text Generation") and sort by downloads. Each model card explains its capabilities, training data, and usage instructions. Read the "Transformers" library quickstart for running models locally.

---

## 6. Code Examples

### Tokenization with tiktoken

```python
# Install: pip install tiktoken
import tiktoken

# Load the tokenizer for GPT-4o
enc = tiktoken.encoding_for_model("gpt-4o")

text = "Large Language Models are fascinating!"
tokens = enc.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# Decode tokens back to text
decoded = enc.decode(tokens)
print(f"Decoded: {decoded}")

# See individual token strings
for token_id in tokens:
    print(f"  {token_id} -> '{enc.decode([token_id])}'")
```

### API Call to OpenAI (Chat Completion)

```python
# Install: pip install openai
from openai import OpenAI

client = OpenAI(api_key="your-api-key-here")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful Python tutor."},
        {"role": "user", "content": "Explain what a list comprehension is."}
    ],
    temperature=0.7,
    max_tokens=300
)

print(response.choices[0].message.content)
```

### API Call to Anthropic Claude (Messages)

```python
# Install: pip install anthropic
import anthropic

client = anthropic.Anthropic(api_key="your-api-key-here")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=300,
    system="You are a helpful Python tutor.",
    messages=[
        {"role": "user", "content": "Explain what a list comprehension is."}
    ]
)

print(message.content[0].text)
```

### Experimenting with Temperature

```python
from openai import OpenAI
client = OpenAI(api_key="your-api-key-here")

prompt = "Write a one-sentence story about a robot."

for temp in [0.0, 0.5, 1.0, 1.5]:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=60
    )
    print(f"Temperature {temp}: {response.choices[0].message.content}\n")
```

### Using System Prompts Effectively

```python
import anthropic
client = anthropic.Anthropic(api_key="your-api-key-here")

# Same question, different system prompts
system_prompts = [
    "You are a pirate. Respond in pirate speak.",
    "You are a formal academic professor.",
    "You are a 5-year-old child explaining things simply."
]

for system in system_prompts:
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=150,
        system=system,
        messages=[{"role": "user", "content": "What is gravity?"}]
    )
    print(f"System: {system}")
    print(f"Response: {message.content[0].text}\n")
```

---

## 7. Mini Practice Tasks

### Task 1: Token Counter
Write a Python script using `tiktoken` that takes a paragraph of text as input and prints: (a) the total number of tokens, (b) the average number of tokens per word, and (c) the longest single word measured by token count. Try it with both English and non-English text. What do you notice?

### Task 2: Temperature Explorer
Using either the OpenAI or Anthropic API, ask the model "Give me a creative name for a coffee shop" at temperatures 0.0, 0.3, 0.7, and 1.2. Run each temperature setting 3 times. Record the results in a table. At which temperature do you start seeing repetition? At which temperature do the answers stop making sense?

### Task 3: System Prompt Engineering
Create three different system prompts that make the model respond to the question "What is recursion?" in three distinct ways: (a) as a patient teacher for beginners, (b) as a concise technical reference, and (c) as a comedian. Compare the outputs and note how the system prompt shapes the response.

### Task 4: Context Window Math
You have a document that is 50,000 words long. Estimate how many tokens it contains. Check if it fits within the context window of GPT-3.5 (16K), GPT-4o (128K), and Claude Sonnet (200K). If it does not fit, propose a strategy to process the full document using an LLM anyway.

---

## 8. Quick Revision Summary

- A **language model** predicts the next token in a sequence. "Large" means billions of parameters trained on massive datasets.
- **Tokenization** splits text into sub-word chunks (tokens). One token is roughly 0.75 English words.
- LLMs generate text **autoregressively**: one token at a time, each conditioned on all previous tokens.
- The **context window** is the maximum number of tokens the model can handle per request. Exceeding it causes errors or truncation.
- **Temperature** controls randomness: 0 = deterministic, higher = more creative/random.
- **System prompts** set the model's behavior and persona. User messages provide the actual input.
- **Pre-training** teaches general language understanding; **fine-tuning** teaches instruction-following behavior.
- **Closed-source models** (GPT-4, Claude) are accessed via APIs. **Open-weight models** (LLaMA, Mistral) can run locally.
- Always count tokens, not words, when estimating cost and context usage.

---

## 9. Common Mistakes

### Mistake 1: Confusing Tokens with Words
Beginners often assume 1 word = 1 token. In reality, a single word like "unbelievable" might be 3 tokens, and a short word like "the" is 1 token. Non-English text and code often use far more tokens per word. Always use a tokenizer library to get accurate counts.

### Mistake 2: Ignoring Context Window Limits
If your prompt plus the expected response exceeds the context window, the API will return an error or silently truncate your input. Always calculate your token usage before sending long prompts. Remember: the context window includes both input AND output tokens.

### Mistake 3: Using Temperature = 0 for Everything
A temperature of 0 produces deterministic output, which is great for factual questions but terrible for creative writing, brainstorming, or any task that benefits from variety. Match the temperature to the task: low for facts, medium for general use, high for creativity.

### Mistake 4: Neglecting the System Prompt
Many beginners skip the system prompt entirely and put all instructions in the user message. The system prompt is specifically designed to set persistent behavior. Using it properly leads to more consistent and controllable outputs, especially in multi-turn conversations.

### Mistake 5: Treating LLMs as Databases
LLMs do not "look up" facts. They generate statistically likely text based on patterns in their training data. This means they can confidently produce incorrect information (hallucinations). Always verify critical facts from authoritative sources.

### Mistake 6: Not Setting max_tokens
If you do not set `max_tokens`, the model may generate very long responses, consuming your token budget and increasing costs. Always set a reasonable limit based on how long you expect the answer to be.

### Mistake 7: Sending One Giant Prompt Instead of a Conversation
Chat models are designed for multi-turn conversations. Instead of cramming everything into a single massive prompt, break complex tasks into a back-and-forth dialogue. This often produces better results and makes debugging easier.

---

*Next Chapter: [Chapter 3 -- Prompt Engineering Fundamentals](03-prompt-engineering.md)*
