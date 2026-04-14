# Chapter 9: Advanced Topics — Agents, Fine-tuning, and Multimodal AI

Now that you have a solid foundation with LangChain, prompt engineering, and RAG, it is time to explore three powerful frontiers of Generative AI: **AI Agents**, **Fine-tuning**, and **Multimodal AI**. Each of these topics unlocks capabilities that take you from building chatbots to building truly intelligent systems.

---

## 1. Concept Explanation

### AI Agents

An AI Agent is an LLM that can **take actions** in the real world. Instead of only generating text, an agent can search the web, query databases, run code, call APIs, and decide *which* tool to use and *when*.

**Analogy:** An LLM is a brain, but an agent has hands — it can Google things, run code, and use tools. A bare LLM can *tell* you how to look up the weather; an agent can actually *go check* the weather API and come back with today's forecast.

### Fine-tuning

Fine-tuning means taking a pre-trained model and training it further on your own dataset so it becomes specialized for a particular domain or task.

**Analogy:** Pre-training teaches a model English. Fine-tuning teaches it *legal English* — or medical English, or your company's specific terminology and style. The model already understands language; you are sharpening it for your use case.

### Multimodal AI

Multimodal models can process and generate more than just text. They handle **images, audio, video, and text** — sometimes all at once.

**Analogy:** Like a person who can see, hear, and read. A text-only model is like communicating through written letters. A multimodal model is like having a face-to-face conversation where you can show pictures, play audio, and point at things.

---

## 2. Detailed Notes

### AI Agents

#### What Are Agents?

An agent is a system where an LLM acts as the "reasoning engine" that decides what actions to take, executes those actions using **tools**, observes the results, and continues until the task is complete. Unlike a simple chain (input -> LLM -> output), an agent operates in a **loop**.

#### The ReAct Pattern

ReAct (Reasoning + Acting) is the most common agent design pattern:

1. **Think** — The LLM reasons about what to do next
2. **Act** — It selects and calls a tool
3. **Observe** — It reads the tool's output
4. **Repeat** — It decides whether to act again or return a final answer

This loop continues until the agent has enough information to answer the user's question.

#### Tool Use

Tools are Python functions that an agent can call. Common examples:
- **Search tools** — Web search, Wikipedia lookup
- **Math tools** — Calculators, Wolfram Alpha
- **Code execution** — Running Python in a sandbox
- **API tools** — Weather, stock prices, databases
- **Custom tools** — Anything you wrap in a function

#### LangChain Agents

LangChain provides `create_react_agent` to build agents quickly. You define tools, pick an LLM, and the framework handles the ReAct loop for you.

#### Agent Types

| Type | Description | Use Case |
|------|-------------|----------|
| ReAct | Reason then act in a loop | General-purpose |
| Plan-and-Execute | Plan all steps first, then execute | Complex multi-step tasks |
| OpenAI Functions | Uses OpenAI's function calling | Structured tool use |
| Multi-agent | Multiple agents collaborate | Complex workflows |

#### Multi-Agent Systems

In advanced setups, multiple agents work together. For example, a "researcher" agent gathers data while a "writer" agent drafts a report. Frameworks like **LangGraph**, **CrewAI**, and **AutoGen** support multi-agent orchestration.

---

### Fine-tuning

#### When to Use What — A Decision Framework

| Approach | When to Use | Cost | Effort |
|----------|-------------|------|--------|
| **Prompt Engineering** | Task can be solved with better instructions | Free | Low |
| **RAG** | Model needs access to your specific data | Low | Medium |
| **Fine-tuning** | Model needs to learn a new style, format, or domain deeply | High | High |

**Rule of thumb:** Start with prompt engineering. If that is not enough, try RAG. Fine-tune only when the model needs to fundamentally change *how* it responds, not just *what* it knows.

#### Types of Fine-tuning

- **Full Fine-tuning** — Update all model weights. Requires massive GPU resources. Rarely done by individuals.
- **LoRA (Low-Rank Adaptation)** — Freeze most weights, train small "adapter" matrices. 10-100x less memory than full fine-tuning.
- **QLoRA (Quantized LoRA)** — Combines 4-bit quantization with LoRA. Fine-tune a 65B parameter model on a single GPU.

#### Data Preparation

Fine-tuning requires high-quality, task-specific data. For OpenAI, data is formatted as JSONL with `messages` arrays. For Hugging Face, you use datasets in various formats. Quality matters far more than quantity — 100 excellent examples often beat 10,000 mediocre ones.

#### Evaluation

After fine-tuning, evaluate your model against a held-out test set. Measure task-specific metrics (accuracy, BLEU score, human preference ratings) and watch for **overfitting** — when the model memorizes training data instead of learning patterns.

---

### Multimodal AI

#### Vision Models

- **GPT-4V / GPT-4o** — Accept images alongside text prompts. Can describe images, read text in photos, analyze charts, and answer visual questions.
- **Claude Vision (Claude 3+)** — Similar capabilities with strong reasoning about image content.
- **Open-source** — LLaVA, Qwen-VL, and others run locally.

#### Image Generation

- **DALL-E 3** — OpenAI's image generation model, accessible via API.
- **Stable Diffusion** — Open-source, runs locally, highly customizable.
- **Midjourney** — Accessible through Discord, popular for artistic output.

#### Audio

- **Whisper** — OpenAI's open-source speech-to-text model. Supports 99 languages. Runs locally or via API.
- **TTS APIs** — OpenAI and others offer text-to-speech with natural-sounding voices.
- **Audio understanding** — GPT-4o can process audio natively.

#### Sending Images to LLM APIs

Both OpenAI and Anthropic accept images as part of the messages array. Images can be sent as **base64-encoded strings** or as **URLs** (OpenAI). The model then reasons about the image content alongside any text prompt.

---

## 3. Visual / Intuitive Explanation

### The ReAct Agent Loop

```
User Question
     |
     v
+---> [THINK] "I need to search for this information"
|         |
|         v
|    [ACT] Call search_tool("query")
|         |
|         v
|    [OBSERVE] "Search returned: ..."
|         |
|         v
|    [THINK] "Now I have enough info to answer"
|         |
+--- OR --+
          |
          v
    [FINAL ANSWER] Return response to user
```

The agent keeps looping through Think-Act-Observe until it decides it has a complete answer. This is what makes agents powerful — they can take *multiple steps* to solve a problem.

### Fine-tuning as Weight Adjustment

```
Pre-trained Model (General Knowledge)
         |
         | + Your Dataset (500-5000 examples)
         |
         v
Fine-tuned Model (Specialized Knowledge)

Think of it like this:
  Base model weights: [0.23, 0.87, 0.45, 0.12, ...]
                       +     -     +     =
  LoRA adjustments:   [0.01, 0.03, 0.02, 0.00, ...]
                       =     =     =     =
  Final weights:      [0.24, 0.84, 0.47, 0.12, ...]

LoRA only learns the small adjustments (deltas), not entire new weights.
```

### Multimodal: Multiple Channels

```
         +-- Text -----> [              ] ----> Text
         |               [              ]
Input ---+-- Image ----> [  Multimodal  ] ----> Image
         |               [    Model     ]
         +-- Audio ----> [              ] ----> Audio

The model has multiple "senses" — it can receive and produce
different types of content through different channels.
```

---

## 4. YouTube Resources (Search Terms)

Search for these on YouTube for high-quality tutorials:

- `"LangChain agents tutorial 2024"` — Building agents with tools step by step
- `"ReAct agent pattern explained"` — Understanding the reasoning-acting loop
- `"Fine-tune LLM with LoRA tutorial"` — Practical fine-tuning walkthroughs
- `"QLoRA explained simply"` — Understanding quantized fine-tuning
- `"GPT-4 Vision API tutorial"` — Sending images to LLMs
- `"OpenAI fine-tuning API tutorial"` — Using the fine-tuning endpoint
- `"Whisper speech to text Python"` — Audio transcription with Whisper

---

## 5. Official Documentation

| Topic | Link |
|-------|------|
| LangChain Agents | https://python.langchain.com/docs/how_to/#agents |
| LangGraph (Multi-agent) | https://langchain-ai.github.io/langgraph/ |
| OpenAI Fine-tuning Guide | https://platform.openai.com/docs/guides/fine-tuning |
| OpenAI Vision Guide | https://platform.openai.com/docs/guides/vision |
| OpenAI Whisper API | https://platform.openai.com/docs/guides/speech-to-text |
| Anthropic Vision Docs | https://docs.anthropic.com/en/docs/build-with-claude/vision |
| Hugging Face PEFT (LoRA) | https://huggingface.co/docs/peft |
| Hugging Face TRL (Training) | https://huggingface.co/docs/trl |

---

## 6. Code Examples

### 6.1 Build a Simple Agent with LangChain

This agent can search the web and perform calculations.

```bash
pip install langchain langchain-openai langchain-community duckduckgo-search
```

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain import hub

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define tools
search = DuckDuckGoSearchRun()

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(name="Search", func=search.run,
         description="Search the web for current information"),
    Tool(name="Calculator", func=calculator,
         description="Evaluate math expressions. Input should be a valid Python math expression."),
]

# Pull the ReAct prompt and create the agent
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
response = agent_executor.invoke({
    "input": "What is the population of France, and what is that number divided by 7?"
})
print(response["output"])
```

When you run this, you will see the agent **think** ("I need to find France's population"), **act** (call the Search tool), **observe** the result, **think** again ("Now I need to divide"), **act** (call Calculator), and then return a final answer.

---

### 6.2 Send an Image to GPT-4V and Claude

**With OpenAI (GPT-4o):**

```bash
pip install openai
```

```python
import openai
import base64

client = openai.OpenAI()

# Option A: Send an image URL
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image? Be detailed."},
                {"type": "image_url", "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
                }},
            ],
        }
    ],
    max_tokens=500,
)
print(response.choices[0].message.content)

# Option B: Send a local image as base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

base64_image = encode_image("photo.jpg")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }},
            ],
        }
    ],
)
print(response.choices[0].message.content)
```

**With Anthropic (Claude):**

```bash
pip install anthropic
```

```python
import anthropic
import base64

client = anthropic.Anthropic()

# Send a local image
with open("photo.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "What is in this image? Describe it in detail."
                }
            ],
        }
    ],
)
print(message.content[0].text)
```

---

### 6.3 Fine-tuning Data Preparation (JSONL Format)

OpenAI expects training data in JSONL format where each line is a conversation:

```python
import json

# Prepare training examples
training_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a customer support agent for Acme Corp. Be concise and helpful."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "Go to acme.com/reset, enter your email, and follow the link sent to your inbox. The link expires in 15 minutes."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a customer support agent for Acme Corp. Be concise and helpful."},
            {"role": "user", "content": "What is your refund policy?"},
            {"role": "assistant", "content": "We offer full refunds within 30 days of purchase. After 30 days, we provide store credit. Contact billing@acme.com to start a refund."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a customer support agent for Acme Corp. Be concise and helpful."},
            {"role": "user", "content": "My order hasn't arrived yet."},
            {"role": "assistant", "content": "I'm sorry to hear that. Standard shipping takes 5-7 business days. Please share your order number and I'll check the tracking status for you."}
        ]
    },
]

# Write to JSONL file
with open("training_data.jsonl", "w") as f:
    for example in training_data:
        f.write(json.dumps(example) + "\n")

print(f"Created {len(training_data)} training examples")

# Validate: each line should be valid JSON
with open("training_data.jsonl", "r") as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            assert "messages" in data
            assert len(data["messages"]) >= 2
            print(f"Example {i+1}: valid ({len(data['messages'])} messages)")
        except Exception as e:
            print(f"Example {i+1}: INVALID - {e}")
```

**Tip:** For a good fine-tune, aim for at least 50-100 high-quality examples. More is better, but quality always beats quantity.

---

### 6.4 OpenAI Fine-tuning API Call

```python
import openai

client = openai.OpenAI()

# Step 1: Upload the training file
upload_response = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)
file_id = upload_response.id
print(f"Uploaded file: {file_id}")

# Step 2: Create a fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file_id,
    model="gpt-4o-mini-2024-07-18",  # Base model to fine-tune
    hyperparameters={
        "n_epochs": 3,  # Number of training passes
    }
)
print(f"Fine-tuning job created: {job.id}")

# Step 3: Monitor progress
import time

while True:
    job_status = client.fine_tuning.jobs.retrieve(job.id)
    print(f"Status: {job_status.status}")
    if job_status.status in ["succeeded", "failed", "cancelled"]:
        break
    time.sleep(60)

# Step 4: Use the fine-tuned model
if job_status.status == "succeeded":
    fine_tuned_model = job_status.fine_tuned_model
    print(f"Fine-tuned model: {fine_tuned_model}")

    response = client.chat.completions.create(
        model=fine_tuned_model,
        messages=[
            {"role": "system", "content": "You are a customer support agent for Acme Corp."},
            {"role": "user", "content": "Can I upgrade my plan?"}
        ]
    )
    print(response.choices[0].message.content)
```

---

### 6.5 Audio Transcription with Whisper

```bash
pip install openai
```

```python
import openai

client = openai.OpenAI()

# Transcribe an audio file
with open("meeting_recording.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text",         # or "json", "srt", "vtt"
        language="en",                   # optional: helps accuracy
    )

print("Transcript:")
print(transcript)

# You can also get timestamps with "verbose_json"
with open("meeting_recording.mp3", "rb") as audio_file:
    detailed = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )

for segment in detailed.segments:
    start = round(segment["start"], 1)
    end = round(segment["end"], 1)
    print(f"[{start}s - {end}s] {segment['text']}")
```

**Running Whisper locally (free, no API key):**

```bash
pip install openai-whisper
```

```python
import whisper

model = whisper.load_model("base")  # tiny, base, small, medium, large
result = model.transcribe("meeting_recording.mp3")
print(result["text"])
```

---

## 7. Mini Practice Tasks

1. **Build a Research Agent** — Create a LangChain agent with a web search tool and a calculator. Ask it: "What is the GDP of Japan in USD, and how much is that per capita?" The agent should search, find the population, and calculate.

2. **Image Analyzer** — Write a script that takes a local image path as input, sends it to GPT-4o (or Claude), and prints a detailed description. Add a follow-up question: "What colors dominate this image?"

3. **Fine-tuning Data Creator** — Pick any domain (cooking, fitness, tech support). Write 20 training examples in JSONL format with a consistent system prompt, user question, and ideal assistant response. Validate the file with the code from Section 6.3.

4. **Transcribe and Summarize** — Use Whisper to transcribe an audio file (find a short podcast clip or record yourself). Then pass the transcript to an LLM and ask it to generate a bullet-point summary.

5. **Multi-tool Agent** — Extend the agent from Task 1 by adding a custom tool that checks the current time (using Python's `datetime` module). Ask the agent: "What time is it in Tokyo right now, and what is a good restaurant to visit there at this hour?"

---

## 8. Quick Revision Summary

| Topic | Key Takeaway |
|-------|-------------|
| AI Agents | LLMs + tools + a reasoning loop. The ReAct pattern: Think, Act, Observe, Repeat. |
| Tools | Python functions that agents can call — search, calculate, query APIs, run code. |
| ReAct Pattern | The agent reasons about what to do, takes an action, observes the result, and repeats. |
| Fine-tuning vs RAG | RAG = give the model new knowledge. Fine-tuning = change how the model behaves. |
| LoRA / QLoRA | Efficient fine-tuning methods that train small adapter weights instead of all parameters. |
| Data for fine-tuning | JSONL format, messages array, minimum 50-100 quality examples. |
| Multimodal input | Send images as base64 or URL in the messages array alongside text. |
| Whisper | OpenAI's speech-to-text model. Runs via API or locally. Supports 99 languages. |
| Vision models | GPT-4o and Claude can analyze images — describe, OCR, answer visual questions. |
| Decision order | Prompt engineering first, then RAG, then fine-tuning. Escalate only when needed. |

---

## 9. Common Mistakes

### Mistake 1: Using fine-tuning when RAG would work

**Wrong thinking:** "My chatbot needs to answer questions about our company docs, so I should fine-tune."
**Why it is wrong:** Fine-tuning teaches style and behavior, not factual recall. For company-specific knowledge, RAG is cheaper, faster, and easier to update. Fine-tune only when you need the model to *behave* differently (tone, format, reasoning style).

### Mistake 2: Not setting `max_iterations` on agents

Agents run in a loop, and without a limit, a confused agent can loop forever — burning through your API budget. Always set `max_iterations` (e.g., 10-15) in your `AgentExecutor`.

```python
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True,
    max_iterations=10,           # Safety limit
    handle_parsing_errors=True,  # Graceful error handling
)
```

### Mistake 3: Sending huge images to vision APIs

Large images increase latency and cost. Most vision APIs resize images internally, so sending a 4000x4000 pixel photo wastes bandwidth. Resize to around 1024px on the longest side before sending.

### Mistake 4: Low-quality fine-tuning data

Garbage in, garbage out. Common data problems:
- Inconsistent formatting across examples
- Factual errors in the "ideal" responses
- Too few examples (under 50)
- No held-out validation set to catch overfitting

Always manually review your training data. Split into training (80%) and validation (20%) sets.

### Mistake 5: Forgetting to handle agent tool errors

If a tool throws an exception, the agent crashes. Always add error handling inside your tool functions and set `handle_parsing_errors=True` on the agent executor.

### Mistake 6: Ignoring cost implications

- Agent calls multiply LLM usage (each Think-Act-Observe cycle is a separate API call)
- Fine-tuning has training costs *and* higher per-token inference costs
- Vision API calls cost more than text-only calls
- Whisper API charges per minute of audio

Always estimate costs before running agents on large tasks or starting a fine-tuning job. Use `gpt-4o-mini` for development and testing.

### Mistake 7: Not specifying the language for Whisper

Whisper auto-detects language, but providing the `language` parameter significantly improves accuracy, especially for non-English audio or audio with mixed languages.

---

**Next Steps:** With agents, fine-tuning, and multimodal AI in your toolkit, you now have the building blocks for production-grade AI applications. The next chapter covers deployment, evaluation, and building real-world projects that tie everything together.
