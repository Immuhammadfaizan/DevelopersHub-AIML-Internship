<div align="center">

# 🧠 DevelopersHub — AI & Machine Learning Internship

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A hands-on journey through the core pillars of AI & ML** — from data visualization and classical machine learning, all the way to deep learning and fine-tuning large language models.

[📊 Task 1](#-task-1--iris-dataset-visualization) · [📈 Task 2](#-task-2--stock-price-prediction) · [❤️ Task 3](#%EF%B8%8F-task-3--heart-disease-prediction) · [💬 Task 4](#-task-4--health-chatbot) · [🧘 Task 5](#-task-5--mental-health-chatbot) · [🏠 Task 6](#-task-6--house-price-prediction)

</div>

---

## 🗂️ Project Overview

This repository contains **6 progressive AI/ML tasks** completed during the DevelopersHub AI & ML Internship. Each task builds upon the skills from the previous one, covering a wide spectrum of real-world machine learning applications.

| # | Task | Domain | ML Paradigm |
|:-:|------|--------|-------------|
| 1 | [Iris Dataset Visualization](#-task-1--iris-dataset-visualization) | Data Science | Exploratory Data Analysis |
| 2 | [Stock Price Prediction](#-task-2--stock-price-prediction) | Finance | Supervised Regression |
| 3 | [Heart Disease Prediction](#%EF%B8%8F-task-3--heart-disease-prediction) | Healthcare | Binary Classification |
| 4 | [Health Chatbot](#-task-4--health-chatbot) | Healthcare / NLP | LLM API Integration |
| 5 | [Mental Health Chatbot](#-task-5--mental-health-chatbot) | Mental Health / NLP | LLM Fine-Tuning |
| 6 | [House Price Prediction](#-task-6--house-price-prediction) | Real Estate | Ensemble Regression |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Immuhammadfaizan/DevelopersHub-AIML-Internship.git
cd DevelopersHub-AIML-Internship

# 2. Create a virtual environment
python -m venv hubenv

# 3. Activate the virtual environment
# Windows
hubenv\Scripts\activate
# macOS / Linux
source hubenv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Running Any Task

```bash
# Open the Jupyter notebook for any task
jupyter notebook task1-dataset-visualization/Iris_Analysis.ipynb

# Or run standalone Python scripts (where available)
python task4-health-chatbot/health_chatbot.py
python task6-house-price-prediction/house_price_prediction.py

# Or launch the Streamlit chatbot app (Task 5)
streamlit run task5-mental-health-chatbot/app.py
```

---

## 📊 Task 1 — Iris Dataset Visualization

> **Goal:** Explore and visualize the classic Iris dataset through comprehensive EDA.

<table>
<tr>
<td width="50%">

### 🔍 What It Does

- Loads the **Iris dataset** (150 samples, 3 species)
- Computes descriptive statistics (mean, std, quartiles)
- Analyzes class distribution across species
- Builds a correlation heatmap of all features
- Creates 7 different visualizations

### 📁 Files

| File | Description |
|------|-------------|
| `Iris_Analysis.ipynb` | Main analysis notebook |
| `data.py` | Dataset loader script |
| `data/iris.csv` | Exported dataset |
| `images/` | All generated plots |

</td>
<td width="50%">

### 📊 Visualizations Created

| # | Visualization |
|:-:|---------------|
| 1 | 🔵 Scatter Plot — Petal Length vs. Width |
| 2 | 📊 Histograms — Feature Distributions |
| 3 | 📈 Seaborn KDE — Sepal Length Density |
| 4 | 📦 Box Plots — Outlier Detection |
| 5 | 🎻 Box Plot by Species — Petal Length |
| 6 | 🔗 Pair Plot — Pairwise Relationships |
| 7 | 🔥 Heatmap — Feature Correlations |

### 🛠️ Tech Stack

`Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn`

</td>
</tr>
</table>

---

## 📈 Task 2 — Stock Price Prediction

> **Goal:** Predict the next day's closing stock price for Apple Inc. (AAPL) using historical market data.

<table>
<tr>
<td width="50%">

### 🔍 What It Does

- Fetches **AAPL stock data** (2020–2025) via the Yahoo Finance API
- Engineers features from Open, High, Low, Close, and Volume
- Trains a **Linear Regression** model on 80/20 split
- Predicts next-day closing prices
- Evaluates with MAE, RMSE, and R² Score
- Exports predictions to CSV

### 📁 Files

| File | Description |
|------|-------------|
| `stock_price_prediction.ipynb` | Full analysis notebook |
| `data/apple_stock_data.csv` | Historical AAPL data |
| `outputs/stock_price_results.csv` | Prediction results |
| `images/` | Trend & prediction plots |

</td>
<td width="50%">

### ⚙️ Pipeline

```
Yahoo Finance API
       ↓
 Data Collection (AAPL 2020-2025)
       ↓
 Feature Engineering (Next_Close shift)
       ↓
 Train/Test Split (80/20)
       ↓
 Linear Regression Model
       ↓
 Predictions + Evaluation
       ↓
 Actual vs. Predicted Plot
```

### 📊 Metrics

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **R²** | Coefficient of Determination |

### 🛠️ Tech Stack

`Pandas` · `NumPy` · `Matplotlib` · `scikit-learn` · `yFinance`

</td>
</tr>
</table>

---

## ❤️ Task 3 — Heart Disease Prediction

> **Goal:** Classify whether a patient is at risk of heart disease based on 14 clinical features.

<table>
<tr>
<td width="50%">

### 🔍 What It Does

- Analyzes the **UCI Heart Disease dataset** (303 patients)
- Performs thorough EDA with 4 visualization types
- Cleans data — handles missing values, removes duplicates
- Trains & compares two classification models
- Evaluates with accuracy, confusion matrix, and ROC-AUC
- Analyzes feature importance
- Saves the best model as a `.pkl` file

### 📁 Files

| File | Description |
|------|-------------|
| `heart_disease_prediction.ipynb` | Full analysis notebook |
| `data/heart_disease_uci.csv` | UCI dataset |
| `outputs/models/heart_disease_model.pkl` | Saved model |
| `images/` | EDA & evaluation plots |

</td>
<td width="50%">

### 🤖 Models Compared

| Model | Purpose |
|-------|---------|
| **Logistic Regression** | Primary classifier |
| **Decision Tree** | Comparison + feature importance |

### 🩺 Clinical Features Used

| Feature | Meaning |
|---------|---------|
| `age` | Patient age |
| `sex` | Gender |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `thalach` | Max heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression |

### 🛠️ Tech Stack

`Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn` · `Joblib`

</td>
</tr>
</table>

---

## 💬 Task 4 — Health Chatbot

> **Goal:** Build an interactive chatbot that answers general health questions using a live LLM API with built-in safety filters.

<table>
<tr>
<td width="50%">

### 🔍 What It Does

- Integrates with **Groq API** for fast LLM inference
- Uses the **Llama 3.1 8B Instruct** model
- Implements careful **prompt engineering** for medically safe responses
- Includes a **keyword-based safety filter** that blocks harmful queries before they reach the API
- Supports two modes: **interactive terminal** and **demo** mode
- Appends a medical disclaimer to every response

### 📁 Files

| File | Description |
|------|-------------|
| `health_chatbot.py` | Main chatbot script |
| `health_chatbot.ipynb` | Notebook walkthrough |
| `screenshots/` | Demo screenshots |

</td>
<td width="50%">

### 🛡️ Safety Features

- **Pre-API Guardrail** — Blocks queries containing keywords like `self-harm`, `overdose`, `suicide`, etc.
- **System Prompt Rules** — The LLM is instructed to never diagnose, never prescribe, and always recommend consulting a professional
- **Mandatory Disclaimer** — Every response includes a health disclaimer

### 🧪 Demo Questions

```
✅ "What are the symptoms of the flu?"
✅ "How can I manage stress naturally?"
✅ "What should I eat for a healthy heart?"
🚫 "How to overdose on medication" → BLOCKED
```

### ⚙️ API Configuration

| Parameter | Value |
|-----------|-------|
| Model | `llama-3.1-8b-instant` |
| Temperature | `0.4` (factual) |
| Max Tokens | `250` (concise) |
| Timeout | `30s` |

### 🛠️ Tech Stack

`Python` · `Groq API` · `Llama 3.1 8B` · `requests`

</td>
</tr>
</table>

---

## 🧘 Task 5 — Mental Health Chatbot

> **Goal:** Build an empathetic mental health support chatbot powered by a locally fine-tuned language model with a premium web UI.

<table>
<tr>
<td width="50%">

### 🔍 What It Does

- Fine-tunes **DistilGPT-2** on the **EmpatheticDialogues** dataset from Facebook AI Research
- Trains locally for 2 epochs on ~1000 examples
- Implements an **Echo Guard** safety layer to catch model repetition
- Deploys via a **premium Streamlit interface** with custom dark mode CSS
- Manages multi-turn conversation history
- Includes 12 diverse empathetic fallback responses

### 📁 Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app (684 lines) |
| `mental_health_chatbot.ipynb` | Fine-tuning notebook |
| `mental_health_chatbot/` | Saved model weights & tokenizer |
| `screenshots/` | App demo screenshots |

</td>
<td width="50%">

### ✨ Premium UI Features

- 🌑 **Dark Mode** — Handcrafted CSS (400+ lines)
- 💬 **Chat Bubbles** — Smooth animated message bubbles
- 🟢 **Status Indicator** — Pulsing dot shows model is active
- 🔠 **Custom Typography** — Inter + Lora from Google Fonts
- ↕️ **Auto-Scroll** — JavaScript-injected smooth scrolling
- 🧹 **Clear Chat** — One-click conversation reset

### 🛡️ Echo Guard System

A known training bug caused the model to echo user input. The Echo Guard:
1. Hashes user input with MD5
2. Deterministically selects from 12 empathetic fallback responses
3. Ensures no echoed responses reach the user

### ⚙️ Generation Config

| Parameter | Value |
|-----------|-------|
| `max_new_tokens` | `80` |
| `temperature` | `0.7` |
| `do_sample` | `True` |

### 🛠️ Tech Stack

`PyTorch` · `Transformers` · `Streamlit` · `Datasets` · `Accelerate`

</td>
</tr>
</table>

---

## 🏠 Task 6 — House Price Prediction

> **Goal:** Predict house sale prices in Ames, Iowa using property features and ensemble learning.

<table>
<tr>
<td width="50%">

### 🔍 What It Does

- Loads the **Kaggle Ames Housing dataset** (1460 houses, 81 features)
- Selects 7 most impactful features from 81
- Handles missing values with median imputation
- Trains a **Gradient Boosting Regressor**
- Achieves **R² > 0.90** with just 7 features
- Visualizes predictions, price distributions, and feature importance

### 📁 Files

| File | Description |
|------|-------------|
| `house_price_prediction.ipynb` | Full analysis notebook |
| `house_price_prediction.py` | Standalone script (79 lines) |
| `data/train.csv` | Kaggle housing dataset |
| `images/` | All generated charts |

</td>
<td width="50%">

### 📋 Selected Features (7 of 81)

| Feature | Description |
|---------|-------------|
| `OverallQual` | Overall quality rating (1–10) |
| `GrLivArea` | Above-ground living area (sq ft) |
| `TotalBsmtSF` | Total basement area (sq ft) |
| `GarageCars` | Garage car capacity |
| `FullBath` | Number of full bathrooms |
| `YearBuilt` | Year originally built |
| `BedroomAbvGr` | Bedrooms above ground |

### 📊 Results

| Metric | Score |
|--------|-------|
| **MAE** | ~$17,000 |
| **RMSE** | ~$26,000 |
| **R²** | **~0.91** ✅ |

### ⚙️ Model Configuration

```python
GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
```

### 🛠️ Tech Stack

`Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `scikit-learn`

</td>
</tr>
</table>

---

## 🧰 Technology Matrix

A bird's-eye view of every tool used across all 6 tasks:

| Technology | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 |
|:-----------|:------:|:------:|:------:|:------:|:------:|:------:|
| Python | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Jupyter Notebook | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Pandas | ✅ | ✅ | ✅ | — | — | ✅ |
| NumPy | ✅ | ✅ | ✅ | — | — | ✅ |
| Matplotlib | ✅ | ✅ | ✅ | — | — | ✅ |
| Seaborn | ✅ | — | ✅ | — | — | ✅ |
| scikit-learn | ✅ | ✅ | ✅ | — | — | ✅ |
| yFinance | — | ✅ | — | — | — | — |
| Joblib | — | — | ✅ | — | — | — |
| Groq API / Llama 3.1 | — | — | — | ✅ | — | — |
| PyTorch | — | — | — | — | ✅ | — |
| Transformers | — | — | — | — | ✅ | — |
| Streamlit | — | — | — | — | ✅ | — |

---

## 📂 Repository Structure

```
DevelopersHub-AIML-Internship/
│
├── 📄 README.md                          ← You are here!
├── 📄 requirements.txt                   ← Global Python dependencies
├── 📁 hubenv/                            ← Python virtual environment
│
├── 📁 task1-dataset-visualization/
│   ├── 📓 Iris_Analysis.ipynb
│   ├── 🐍 data.py
│   ├── 📁 data/iris.csv
│   └── 📁 images/ (7 plots)
│
├── 📁 task2-predict-future-stock/
│   ├── 📓 stock_price_prediction.ipynb
│   ├── 📁 data/apple_stock_data.csv
│   ├── 📁 outputs/stock_price_results.csv
│   └── 📁 images/ (2 plots)
│
├── 📁 task3-heart-disease-prediction/
│   ├── 📓 heart_disease_prediction.ipynb
│   ├── 📁 data/heart_disease_uci.csv
│   ├── 📁 outputs/models/heart_disease_model.pkl
│   └── 📁 images/ (7 plots)
│
├── 📁 task4-health-chatbot/
│   ├── 📓 health_chatbot.ipynb
│   ├── 🐍 health_chatbot.py
│   └── 📁 screenshots/
│
├── 📁 task5-mental-health-chatbot/
│   ├── 📓 mental_health_chatbot.ipynb
│   ├── 🐍 app.py
│   ├── 📁 mental_health_chatbot/ (model weights)
│   └── 📁 screenshots/
│
└── 📁 task6-house-price-prediction/
    ├── 📓 house_price_prediction.ipynb
    ├── 🐍 house_price_prediction.py
    ├── 📁 data/train.csv
    └── 📁 images/ (4 plots)
```

---

## 📈 Learning Progression

This internship follows a carefully structured path through the AI/ML landscape:

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │   TASK 1     │     │   TASK 2     │     │   TASK 3     │
 │  📊 EDA &    │────▶│  📈 Regression│────▶│  ❤️ Binary   │
 │ Visualization│     │  (Time-Series)│     │ Classification│
 └──────────────┘     └──────────────┘     └──────────────┘
                                                   │
        ┌──────────────────────────────────────────┘
        ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │   TASK 4     │     │   TASK 5     │     │   TASK 6     │
 │  💬 NLP      │────▶│  🧠 LLM      │────▶│  🏠 Ensemble │
 │  (API-Based) │     │ Fine-Tuning  │     │  Regression  │
 └──────────────┘     └──────────────┘     └──────────────┘
```

| Phase | Skills Gained |
|-------|--------------|
| **Foundation** (Task 1) | Data loading, statistical analysis, visualization best practices |
| **Prediction** (Tasks 2 & 3) | Feature engineering, model training, evaluation metrics, model comparison |
| **NLP & LLMs** (Tasks 4 & 5) | Prompt engineering, API integration, transformer fine-tuning, safety filtering |
| **Applied ML** (Task 6) | End-to-end ML pipeline, ensemble methods, feature selection from large datasets |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to:

1. **Fork** this repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📬 Contact

**Muhammad Faizan** — DevelopersHub AI/ML Intern

[![GitHub](https://img.shields.io/badge/GitHub-Immuhammadfaizan-181717?style=flat-square&logo=github)](https://github.com/Immuhammadfaizan)

---

<div align="center">

**⭐ If you found this repository helpful, please consider giving it a star! ⭐**

*Built with ❤️ during the DevelopersHub AI & ML Internship*

</div>
