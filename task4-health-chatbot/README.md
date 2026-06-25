# Task 4: General Health Query Chatbot

**DevelopersHub Corp — AI/ML Internship**

---

## Objective

Build a chatbot that answers general health questions using a Large Language Model, with a focus on prompt engineering and safety filtering.

---

## Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Groq API | LLM inference (free, fast) |
| Llama 3.1 8B Instruct | Language model |
| Jupyter Notebook | Development and demonstration |
| VS Code | Editor |
| `requests` | HTTP API calls |
| `textwrap` | Response formatting |

---

## Project Structure

```
task4/
│
├── health_chatbot.py       # Main Python script (terminal use)
├── health_chatbot.ipynb    # Jupyter Notebook (full walkthrough)
├── README.md               # This file
└── screenshots/
    ├── sore-throat.png          # Q1 — What causes a sore throat?
    ├── paracetamol.png          # Q2 — Is paracetamol safe for children?
    ├── Additional-Test-Queries  # Q3–Q5 — Additional health queries
    ├── fever.png                # Q6 — How can I reduce a mild fever?
    ├── safety-filters.png       # Safety filter blocking harmful query
    └── terminal-run.png         # Interactive terminal mode
```

---

## How It Works

### 1. Prompt Engineering

The core of this task is the `SYSTEM_PROMPT` — a carefully written instruction block that defines the chatbot's behavior before the user's question is sent to the model.

```
You are a helpful and friendly medical information assistant...
```

It controls:
- The AI's **role and persona**
- **Rules** it must follow (no diagnosis, no prescriptions)
- **Output format** (3–5 sentences)
- **Mandatory disclaimer** at the end of every response

### 2. Safety Filter

A keyword-based guardrail runs before every API call. If the user's question contains any blocked term, the function returns a safe refusal message instantly — no API call is made.

Blocked keywords include: `overdose`, `self-harm`, `suicide`, `kill myself`, `illegal drug`, and others.

### 3. API Call

Questions that pass the safety filter are sent to the **Groq API** using the `llama-3.1-8b-instant` model. The request uses:
- `temperature: 0.4` — factual, consistent responses
- `max_tokens: 250` — concise output (3–5 sentences)

---

## Setup & Usage

### Step 1 — Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up free (Google login works)
3. Click **API Keys** → **Create API Key**
4. Copy the key (starts with `gsk_...`)

### Step 2 — Install Dependencies

```bash
pip install requests
```

### Step 3 — Add Your API Key

Open `health_chatbot.py` or the notebook and replace:

```python
GROQ_API_KEY = "gsk_your_key_here"
```

### Step 4 — Run

**Terminal (interactive mode):**
```bash
python health_chatbot.py
```

**Terminal (demo mode — runs all test queries):**
```bash
python health_chatbot.py --demo
```

**Notebook:**
Open `health_chatbot.ipynb` in VS Code or Jupyter and run all cells top to bottom.

---

## Example Queries & Responses

**Q: What causes a sore throat?**
> A sore throat can be caused by a variety of things, including viruses, bacteria, and allergies. Common culprits include the common cold and flu...
> Please consult a healthcare professional for personalized medical advice.

**Q: Is paracetamol safe for children?**
> Paracetamol can be safe for children, but it's essential to use it carefully and follow the recommended dosage...
> Please consult a healthcare professional for personalized medical advice.

---

## Safety Filter Demo

**Query:** `How do I overdose on medication?`

**Response:**
> I'm not able to assist with that query. If you or someone you know is in distress, please reach out to a healthcare professional or a mental health helpline immediately.

---

## Screenshots

| File | Description |
|---|---|
| `sore-throat.png` | Response to "What causes a sore throat?" |
| `paracetamol.png` | Response to "Is paracetamol safe for children?" |
| `Additional-Test-Queries` | Additional health question responses |
| `fever.png` | Response to fever-related query |
| `safety-filters.png` | Safety filter blocking a harmful query |
| `terminal-run.png` | Interactive terminal session |

---

## Key Concepts Demonstrated

- **Prompt Engineering** — System prompt design controlling role, rules, tone, and format
- **LLM API Integration** — Groq API with Llama 3.1 via HTTP requests
- **Safety Handling** — Pre-API keyword filter as a guardrail layer
- **Conversational Agent** — Interactive loop with clean input/output handling

---

*Muhammad Faizan | AI/ML Engineering Intern | DevelopersHub Corp*
