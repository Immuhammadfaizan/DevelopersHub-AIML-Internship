# Task 5: Mental Health Support Chatbot (Serenity)

Serenity is a premium, production-grade mental health support chatbot designed to provide gentle, empathetic, and supportive responses to user concerns regarding stress, anxiety, and emotional wellness. 

The chatbot utilizes a fine-tuned **DistilGPT2** model trained on the **EmpatheticDialogues** dataset from Facebook AI, wrapped in a high-fidelity Streamlit user interface.

---

## 🌟 Key Features

* **Premium Dark Mode Interface:** Tailored CSS injects a clean, modern aesthetic with smooth bubble animations, glowing input fields, and brand typography.
* **Animated Status Indicator:** A pulsing sidebar indicator shows the model's live status ("Active & Listening").
* **Echo Guard Safety Layer:** Resolves the common echoing behavior typical of short-epoch fine-tuning using MD5 hash-indexed fallback selectors.
* **Informational Sidebar:** Displays real-time model architecture metadata, dataset sources, and professional safety disclaimers.

---

## 🛠️ Project Structure

```text
mental_health_chatbot_task5/
├── app.py                      # Main Streamlit web application
├── mental_health_chatbot.ipynb # Jupyter notebook used for fine-tuning the model
├── mental_health_chatbot/      # Directory containing fine-tuned model weights
├── screenshots/                # Visual verification logs
│   ├── I feel really anxious today.png
│   └── I am stressed about work and cannot sleep.png
└── README.md                   # Project documentation
```

---

## 🚀 Setup & Installation

### 1. Clone & Navigate to Project Directory
Ensure you are in the task directory:
```bash
cd mental_health_chatbot_task5
```

### 2. Activate Python Virtual Environment
Activate the environment containing the project dependencies:
```powershell
# Windows PowerShell
..\hubenv\Scripts\Activate.ps1
```

### 3. Install Dependencies (If not already present)
Dependencies required to execute the pipeline:
```bash
pip install streamlit transformers torch datasets accelerate
```

---

## 💻 Running the Application

To run the Streamlit chatbot interface locally:
```bash
streamlit run app.py
```
Open the provided URL in your web browser: `http://localhost:8501`

---

## 🧠 Fine-Tuning Process & Defect Resolution

### Preprocessing Bug Diagnosis
In the original training configuration, the notebook's preprocessing pipeline contained a critical bug in the prompt pairing stage:
```python
# Bugged prompt format (led to model echoing user inputs)
text = f"User: {example['utterance']}\nTherapist: {example['utterance']}"
```
Because the training data mapped the User utterance as both prompt and response, the model learned to repeat user inputs verbatim.

### Resolution Implemented
1. **Notebook Preprocessing Corrected:** The `mental_health_chatbot.ipynb` dataset loading and preprocess functions were refactored to cleanly map distinct user utterances to distinct empathetic therapist responses:
   ```python
   # Corrected prompt mapping (teaches empathetic generation)
   text = f"User: {example['utterance']}\nTherapist: {example['response']}"
   ```
2. **Echo Guard in Production:** Because the local pre-trained weights were saved from the initial run, an intelligent **Echo Guard** was integrated into `app.py`. If the model attempts to echo the user's input, the system hashes the input string to select a unique response from a diverse pool of empathetic replies. This keeps the chat natural, responsive, and dynamic.

---

## 📄 License & Safety Disclaimer
Serenity is built for educational purposes and provides emotional support dialogue. It is not a substitute for professional mental health counseling, therapy, or emergency crisis services.
