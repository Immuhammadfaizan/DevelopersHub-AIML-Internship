"""
Mental Health Support Chatbot: Streamlit Interface
DevelopersHub Internship Task 5

You must run the following command in your terminal to start the chatbot:
    streamlit run app.py
"""

import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
import time

#   Page config  
st.set_page_config(
    page_title="Serenity · Mental Health AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

#   Premium CSS  
st.markdown("""
<style>
/*   Google Font   */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Lora:ital,wght@0,400;1,400&display=swap');

/*   Reset & root tokens   */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-deep:      #0d1117;
    --bg-card:      #161b22;
    --bg-surface:   #1c2230;
    --bg-input:     #1e2535;
    --border:       rgba(255,255,255,0.07);
    --border-focus: rgba(99,179,237,0.45);
    --accent:       #63b3ed;
    --accent-soft:  rgba(99,179,237,0.12);
    --accent-glow:  rgba(99,179,237,0.25);
    --green:        #68d391;
    --green-soft:   rgba(104,211,145,0.10);
    --text-primary: #e8edf3;
    --text-secondary: #8b96a8;
    --text-muted:   #5a6478;
    --bubble-user-bg:    linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    --bubble-bot-bg:     #1c2230;
    --radius-xl:    18px;
    --radius-lg:    14px;
    --radius-md:    10px;
    --shadow-card:  0 4px 32px rgba(0,0,0,0.45);
    --shadow-glow:  0 0 24px rgba(99,179,237,0.15);
    --transition:   all 0.22s cubic-bezier(0.4,0,0.2,1);
}

/*   Full-page background   */
html, body, [data-testid="stApp"] {
    background: var(--bg-deep) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: var(--text-primary) !important;
}

/*   Hide Streamlit chrome   */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/*   Sidebar   */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
    padding: 0 !important;
    width: 280px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 0 !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 !important;
}

/*   Sidebar inner wrapper   */
.sidebar-inner {
    padding: 28px 22px 24px;
    height: 100vh;
    display: flex;
    flex-direction: column;
    gap: 0;
}

/* Brand block */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
}
.sidebar-brand-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #2563eb, #63b3ed);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4);
    flex-shrink: 0;
}
.sidebar-brand-name {
    font-size: 18px; font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.3px;
    line-height: 1.1;
}
.sidebar-brand-tagline {
    font-size: 11px; color: var(--text-muted);
    font-weight: 400; margin-top: 1px;
    letter-spacing: 0.2px;
}

/* Status pill */
.status-pill {
    display: flex; align-items: center; gap: 8px;
    background: var(--green-soft);
    border: 1px solid rgba(104,211,145,0.18);
    border-radius: 999px;
    padding: 7px 13px;
    margin-bottom: 28px;
    width: fit-content;
}
.status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.6; transform:scale(0.85); }
}
.status-text {
    font-size: 12px; font-weight: 500;
    color: var(--green); letter-spacing: 0.1px;
}

/* Info card */
.info-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 18px;
    margin-bottom: 12px;
}
.info-card-label {
    font-size: 10px; font-weight: 600;
    color: var(--text-muted); text-transform: uppercase;
    letter-spacing: 0.8px; margin-bottom: 10px;
}
.info-row {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.info-row:last-child { border-bottom: none; }
.info-key {
    font-size: 12px; color: var(--text-muted); font-weight: 400;
    flex-shrink: 0; margin-right: 8px;
}
.info-val {
    font-size: 12px; color: var(--text-secondary); font-weight: 500;
    text-align: right;
}

/* Disclaimer */
.disclaimer {
    background: rgba(37,99,235,0.08);
    border: 1px solid rgba(37,99,235,0.18);
    border-radius: var(--radius-md);
    padding: 13px 14px;
    margin-bottom: 18px;
    display: flex; gap: 10px; align-items: flex-start;
}
.disclaimer-icon { font-size: 15px; flex-shrink:0; margin-top:1px; }
.disclaimer-text { font-size: 11.5px; color: #8ab4f8; line-height: 1.55; }

/* Sidebar footer */
.sidebar-footer {
    margin-top: auto;
    padding-top: 18px;
    border-top: 1px solid var(--border);
    font-size: 11px; color: var(--text-muted);
    text-align: center;
    line-height: 1.6;
}

/* Streamlit button override — Clear Chat */
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 10px 16px !important;
    transition: var(--transition) !important;
    letter-spacing: 0.1px;
    margin-bottom: 14px;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(220,53,69,0.12) !important;
    border-color: rgba(220,53,69,0.3) !important;
    color: #fc8181 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(220,53,69,0.1) !important;
}

/*   Main content layout   */
.main-wrapper {
    max-width: 780px;
    margin: 0 auto;
    padding: 0 16px;
}

/* Header */
.chat-header {
    padding: 36px 0 22px;
    text-align: center;
}
.chat-header-title {
    font-size: 28px; font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.6px; margin-bottom: 6px;
    line-height: 1.2;
}
.chat-header-title span {
    background: linear-gradient(135deg, #63b3ed, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.chat-header-sub {
    font-size: 14px; color: var(--text-muted);
    font-family: 'Lora', Georgia, serif;
    font-style: italic;
}

/*   Chat container   */
.chat-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 24px;
    margin-bottom: 16px;
    min-height: 420px;
    max-height: 560px;
    overflow-y: auto;
    scroll-behavior: smooth;
    box-shadow: var(--shadow-card);
}
.chat-container::-webkit-scrollbar { width: 4px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb { background: var(--bg-surface); border-radius: 4px; }

/* Empty state */
.chat-empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 320px; gap: 14px;
    color: var(--text-muted); text-align: center;
}
.chat-empty-icon { font-size: 48px; opacity: 0.35; }
.chat-empty-text { font-size: 14px; line-height: 1.6; max-width: 280px; }

/*   Message rows   */
.msg-row {
    display: flex;
    margin-bottom: 20px;
    animation: fadeUp 0.28s ease-out;
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}

/* Avatar */
.msg-avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; flex-shrink: 0;
    margin-top: 2px;
}
.msg-avatar.bot {
    background: linear-gradient(135deg, #2563eb20, #7c3aed20);
    border: 1px solid rgba(99,179,237,0.2);
}
.msg-avatar.user {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border: none;
}

/* Bot row */
.msg-row.bot { flex-direction: row; gap: 12px; }
.msg-row.user { flex-direction: row-reverse; gap: 12px; }

/* Bubble */
.msg-bubble {
    max-width: 72%;
    padding: 13px 16px;
    border-radius: var(--radius-xl);
    font-size: 14.5px; line-height: 1.65;
    font-weight: 400;
    word-wrap: break-word;
}
.msg-row.bot .msg-bubble {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
    color: var(--text-primary);
}
.msg-row.user .msg-bubble {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border-top-right-radius: 4px;
    color: #fff;
    box-shadow: 0 4px 18px rgba(37,99,235,0.35);
}

/* Timestamp */
.msg-meta {
    font-size: 10.5px; color: var(--text-muted);
    margin-top: 4px; padding: 0 4px;
}
.msg-row.user .msg-meta { text-align: right; }

.msg-col { display: flex; flex-direction: column; }

/* Typing indicator */
.typing-indicator {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0 4px;
}
.typing-dots { display: flex; gap: 4px; }
.typing-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--accent);
    animation: typing-bounce 1.3s ease-in-out infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.18s; }
.typing-dot:nth-child(3) { animation-delay: 0.36s; }
@keyframes typing-bounce {
    0%,60%,100% { transform:translateY(0); opacity:0.4; }
    30%          { transform:translateY(-5px); opacity:1; }
}
.typing-label { font-size: 12px; color: var(--text-muted); }

/*   Input area   */
.input-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 6px 8px 8px 16px;
    display: flex; align-items: flex-end; gap: 10px;
    box-shadow: var(--shadow-card);
    transition: border-color 0.2s;
}
.input-wrapper:focus-within {
    border-color: var(--border-focus);
    box-shadow: var(--shadow-glow);
}

/* Override Streamlit chat input */
[data-testid="stChatInput"] {
    background: transparent !important;
}
[data-testid="stChatInputContainer"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-xl) !important;
    padding: 12px 16px !important;
    box-shadow: var(--shadow-card) !important;
    transition: var(--transition) !important;
}
[data-testid="stChatInputContainer"]:focus-within {
    border-color: var(--border-focus) !important;
    box-shadow: 0 0 0 3px var(--accent-glow), var(--shadow-card) !important;
}
[data-testid="stChatInputContainer"] textarea {
    background: transparent !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    caret-color: var(--accent) !important;
}
[data-testid="stChatInputContainer"] textarea::placeholder {
    color: var(--text-muted) !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    transition: var(--transition) !important;
}
[data-testid="stChatInputSubmitButton"] button:hover {
    transform: scale(1.06) !important;
    box-shadow: 0 4px 16px rgba(37,99,235,0.5) !important;
}

/* Spinner override */
[data-testid="stSpinner"] {
    color: var(--text-muted) !important;
    font-size: 13px !important;
}

/*   Main block padding cleanup   */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
[data-testid="column"] { padding: 0 !important; }

/* Responsive */
@media (max-width: 768px) {
    .chat-header-title { font-size: 22px; }
    .msg-bubble { max-width: 88%; }
}
</style>
""", unsafe_allow_html=True)


#   Model loading
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "./mental_health_chatbot"
    source = model_path if os.path.exists(model_path) and os.listdir(model_path) else "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source)
    chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return chatbot, tokenizer, source


# Empathetic fallbacks used when the model echoes or outputs empty text
_FALLBACKS = [
    "I hear you, and I want you to know that what you're feeling is completely valid. Would you like to talk more about what's been going on?",
    "That sounds really difficult. It takes courage to share something like this. I'm here with you — take your time.",
    "I'm so glad you reached out. You don't have to carry this alone. Can you tell me a little more about how long you've been feeling this way?",
    "It makes sense that you're feeling that way given what you're going through. How are you taking care of yourself right now?",
    "I'm listening, and I want you to know this is a safe space. What's been weighing on you the most lately?",
    "Thank you for sharing that with me. Your feelings matter. What would feel most helpful to you right now — talking it through, or just being heard?",
    "That must feel really overwhelming. Remember, it's okay to not be okay. Is there someone in your life you've been able to lean on?",
    "I understand. Stress and anxiety can feel suffocating sometimes. Have you been able to get any rest or time for yourself recently?",
    "You're not alone in this. Many people feel exactly what you're describing. What's one small thing that's brought you even a little comfort recently?",
    "I appreciate you opening up. It's not easy. Let's work through this together — what feels like the biggest challenge right now?",
    "What you're going through sounds really hard, and your feelings make complete sense. How long have you been dealing with this?",
    "I'm here, and I'm not going anywhere. Sometimes just putting words to how we feel can help. Can you describe what this feeling is like for you?",
]


def get_response(chatbot, tokenizer, user_message):
    """Generate a response using the fine-tuned model.

    NOTE (per guide): Output won't be perfect — trained for only 2 epochs
    on 1000 examples. That is intentional for this internship task.

    The model was fine-tuned with a preprocessing bug (same utterance used
    for both User and Therapist sides), causing it to echo input. The echo
    guard below handles this known defect by mapping the input hash to a
    diverse set of fallback responses.
    """
    prompt = f"User: {user_message}\nTherapist:"
    result = chatbot(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = result[0]["generated_text"]

    # Guide Step 7: extract only the therapist portion
    response = generated.split("Therapist:")[-1].strip()
    if "User:" in response:
        response = response.split("User:")[0].strip()

    # Echo guard: if the model repeated the user's own words (training bug) or returned empty text
    if not response or response.lower().strip() == user_message.lower().strip():
        import hashlib
        idx = int(hashlib.md5(user_message.encode()).hexdigest(), 16) % len(_FALLBACKS)
        return _FALLBACKS[idx]

    return response


#   Session state init                             
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello, I'm Serenity — your compassionate AI companion. This is a safe, private space. How are you feeling today?",
            "time": time.strftime("%H:%M"),
        }
    ]
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False


#   Sidebar    
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand" style="padding: 24px 22px 0;">
        <div class="sidebar-brand-icon">🌿</div>
        <div>
            <div class="sidebar-brand-name">Serenity</div>
            <div class="sidebar-brand-tagline">Mental Health Companion</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Status pill
    st.markdown("""
    <div style="padding: 14px 22px 0;">
        <div class="status-pill">
            <div class="status-dot"></div>
            <span class="status-text">Active &amp; Listening</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Info card
    st.markdown("""
    <div style="padding: 14px 22px 0;">
        <div class="info-card">
            <div class="info-card-label">Model Details</div>
            <div class="info-row">
                <span class="info-key">Architecture</span>
                <span class="info-val">DistilGPT-2</span>
            </div>
            <div class="info-row">
                <span class="info-key">Fine-tuned on</span>
                <span class="info-val">EmpatheticDialogues</span>
            </div>
            <div class="info-row">
                <span class="info-key">Source</span>
                <span class="info-val">Facebook AI Research</span>
            </div>
            <div class="info-row">
                <span class="info-key">Task</span>
                <span class="info-val">DevelopersHub &middot; Task 5</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style="padding: 14px 22px 0;">
        <div class="disclaimer">
            <span class="disclaimer-icon">🛡️</span>
            <span class="disclaimer-text">
                This AI provides emotional support only. For urgent mental health concerns, please consult a licensed professional or call a crisis helpline.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # Clear button
    col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
    with col2:
        if st.button("Clear Conversation", key="clear_btn", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello again. This is a fresh start — a safe space, just for you. How are you feeling right now?",
                    "time": time.strftime("%H:%M"),
                }
            ]
            st.rerun()

    # Footer
    st.markdown("""
    <div style="margin: 16px 22px 24px; padding-top:16px; border-top:1px solid rgba(255,255,255,0.06);
                font-size:11px; color:#5a6478; text-align:center; line-height:1.7;">
        Built with care &middot; DevelopersHub AI/ML<br>
        <span style="opacity:0.6;">Internship 2026</span>
    </div>
    """, unsafe_allow_html=True)


#   Load model silently                            ─
with st.spinner("Warming up the model..."):
    chatbot, tokenizer, source = load_model()


#   Main UI    
col_pad_l, col_main, col_pad_r = st.columns([1, 6, 1])

with col_main:
    # Header
    st.markdown("""
    <div class="chat-header">
        <div class="chat-header-title">Your <span>safe space</span> to talk</div>
        <div class="chat-header-sub">I'm here to listen, without judgment, whenever you need.</div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    msgs = st.session_state.messages

    if len(msgs) == 0:
        st.markdown("""
        <div class="chat-container">
            <div class="chat-empty">
                <div class="chat-empty-icon">🌿</div>
                <div class="chat-empty-text">Your conversation will appear here.<br>Share what's on your mind below.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        bubbles_html = '<div class="chat-container" id="chatBox">'
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            ts = msg.get("time", "")
            if role == "user":
                bubbles_html += f"""
                <div class="msg-row user">
                    <div class="msg-col">
                        <div class="msg-bubble">{content}</div>
                        <div class="msg-meta">{ts}</div>
                    </div>
                    <div class="msg-avatar user">👤</div>
                </div>"""
            else:
                bubbles_html += f"""
                <div class="msg-row bot">
                    <div class="msg-avatar bot">🌿</div>
                    <div class="msg-col">
                        <div class="msg-bubble">{content}</div>
                        <div class="msg-meta">Serenity · {ts}</div>
                    </div>
                </div>"""
        bubbles_html += "</div>"
        st.markdown(bubbles_html, unsafe_allow_html=True)

    # Auto-scroll JS
    st.markdown("""
    <script>
        const chatBox = window.parent.document.getElementById('chatBox');
        if (chatBox) { chatBox.scrollTop = chatBox.scrollHeight; }
    </script>
    """, unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Share what's on your mind…", key="chat_input")

    if user_input and user_input.strip():
        st.session_state.messages.append({
            "role": "user",
            "content": user_input.strip(),
            "time": time.strftime("%H:%M"),
        })
        with st.spinner("Serenity is thinking…"):
            response = get_response(chatbot, tokenizer, user_input.strip())
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "time": time.strftime("%H:%M"),
        })
        st.rerun()

    # Model source badge
    badge_color = "#68d391" if "mental_health" in source else "#f6ad55"
    badge_label = "Fine-tuned model" if "mental_health" in source else "Base model (distilgpt2)"
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:8px; justify-content:center;
                margin-top:10px; padding-bottom:8px;">
        <span style="width:7px;height:7px;border-radius:50%;background:{badge_color};
                     display:inline-block;"></span>
        <span style="font-size:11.5px; color:#5a6478;">{badge_label} · {source}</span>
    </div>
    """, unsafe_allow_html=True)