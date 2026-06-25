"""
What this script does:
  1. Accepts a health question from the user.
  2. Runs it through a safety filter (blocks harmful queries).
  3. Wraps the question in a carefully engineered system prompt.
  4. Sends it to Llama 3  via Groq API.
  5. Returns a clear, friendly, safe health response.
"""

import requests

# API TOKEN Configuration
GROQ_API_KEY = "" # Replace with your actual Groq API key - due to some security reasons, the key is not included here. You must set it in your environment or directly in the code. 

# check the screenshots for more information

API_URL = "https://api.groq.com/openai/v1/chat/completions"


# System prompt to guide AI  behavior

SYSTEM_PROMPT = """You are a helpful and friendly medical information assistant.
Your job is to answer general health questions in simple, clear language that anyone can understand.

IMPORTANT RULES you must always follow:
- Always recommend seeing a doctor for serious, persistent, or emergency symptoms.
- Never diagnose any disease or medical condition.
- Never prescribe or recommend specific medications or dosages.
- Keep answers concise: 3 to 5 sentences maximum.
- Use simple, everyday language, and avoid complex medical jargon.
- If a question asks about something dangerous, illegal, or harmful, politely decline.
- Always end every response with exactly this line:
  "Please consult a healthcare professional for personalized medical advice."

Your tone should be warm, calm, and reassuring like a knowledgeable friend, not a cold textbook."""


# Safety keywords to block harmful queries

BLOCKED_KEYWORDS = [
    "overdose",
    "self-harm",
    "self harm",
    "suicide",
    "kill myself",
    "illegal drug",
    "how to harm",
    "poison someone",
]


def safety_filter(user_question: str) -> str | None:
    """
    Check if the user's question contains any blocked content.

    Returns:
        A safe refusal message (str) if blocked.
        None if the question is safe to proceed.
    """
    question_lower = user_question.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in question_lower:
            return (
                "I'm not able to assist with that query. "
                "If you or someone you know is in distress, please reach out to a "
                "healthcare professional or a mental health helpline immediately."
            )
    return None  # Safe to proceed


# Main chatbot function - runs filters, builds prompt and sends request to Groq.

def ask_health_question(user_question: str) -> str:
    """
    Send a health question to Llama 3 and return the response.

    Args:
        user_question: The health question typed by the user.

    Returns:
        The AI's response as a clean string.
    """
    # safety check
    blocked_response = safety_filter(user_question)
    if blocked_response:
        return blocked_response

    # prepare the API request
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_question},
        ],
        "temperature": 0.4,  # lower provides more factual responses
        "max_tokens": 250,   # AI response maximum length
    }

    # make the API call
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            return f"API Error {response.status_code}: {response.text[:200]}"

    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"Connection error: {str(e)}"


# Example queries for general  testing

DEMO_QUESTIONS = [
    "What causes a sore throat?",
    "Is paracetamol safe for children?",
    "How much water should I drink per day?",
    "What are the symptoms of dehydration?",
    "How can I reduce a mild fever at home?",
    "What foods are good for a healthy heart?",
    "Is it normal to feel tired all the time?",
]


def run_demo():
    """Run the chatbot on all demo questions and print results."""
    print("\n" + "=" * 65)
    print("HEALTH QUERY CHATBOT — DEMO MODE")
    print(": Llama 3 via Groq API")
    print("=" * 65 + "\n")

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"[Q{i}] {question}")
        print("-" * 50)
        answer = ask_health_question(question)
        print(f"Bot: {answer}")
        print("=" * 65 + "\n")


# Interactive loop — type 'quit' or 'exit' to stop

def run_interactive():
    """Start an interactive chatbot session in the terminal."""
    print("\n" + "=" * 65)
    print("HEALTH QUERY CHATBOT — Interactive Mode")
    print("Type your health question and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 65 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! Stay healthy!")
            break

        if not user_input:
            print("(Please type a question.)\n")
            continue

        if user_input.lower() in ["quit", "exit", "bye", "q"]:
            print("Goodbye! Stay healthy!")
            break

        response = ask_health_question(user_input)
        print(f"{response}\n")
        print("-" * 65 + "\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        run_interactive()