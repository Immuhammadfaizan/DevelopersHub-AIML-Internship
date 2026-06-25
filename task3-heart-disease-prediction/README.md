# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a person is at risk of heart disease using health-related medical attributes from the **Heart Disease UCI Dataset**.

The project follows a complete Machine Learning workflow, including:

- Data Loading and Inspection
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Model Evaluation
- ROC-AUC Analysis
- Feature Importance Analysis
- Model Saving

The final model helps classify whether a patient is likely to have heart disease based on their health information.

---

# 🎯 Objective

Build a machine learning model to predict whether a person is at risk of heart disease based on their health data.

---

# 📊 Dataset

**Dataset Name:** Heart Disease UCI Dataset

**Source:** Kaggle

The dataset contains various medical attributes such as:

| Feature | Description |
|----------|-------------|
| age | Age of patient |
| sex | Gender |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Cholesterol level |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |
| target | Heart disease presence |

---

# 🛠 Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib

---

# 📂 Project Structure

```text
task3-heart-disease-prediction/
│
├── data/
│   └── heart_disease_uci.csv
│
├── images/
│   ├── chest_pain_vs_heart_disease.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── gender_vs_heart_disease.png
│   ├── heart_disease_distribution.png
│   └── roc_curve.png
│
├── outputs/
│   └── models/
│       └── heart_disease_model.pkl
│
├── screenshots/
│   ├── AUC_score.png
│   ├── available_and_final_column.png
│   ├── dataset_describe.png
│   ├── decision_tree_accuracy.png
│   ├── features_importance.png
│   ├── handling_missing_values.png
│   ├── loading_dataset_result.png
│   ├── model_saving.png
│   └── model_training_evaluation.png
│
├── heart_disease_prediction.ipynb
│
└── README.md
```

---

# 🚀 Project Workflow

## Step 1: Data Loading

The dataset is loaded using Pandas and inspected to understand its structure.

### Tasks Performed

- Read CSV file
- Display first few rows
- Check dataset dimensions
- Inspect data types

---

## Step 2: Data Cleaning

Data quality checks are performed before model training.

### Tasks Performed

- Check missing values
- Handle missing values if present
- Remove duplicate records
- Standardize column names
- Verify target column

### Why This Step?

Machine learning models perform better when trained on clean and consistent data.

---

## Step 3: Exploratory Data Analysis (EDA)

EDA helps understand patterns and relationships within the dataset.

### Visualizations Created

#### Heart Disease Distribution

Shows the number of patients with and without heart disease.

**Output:**

```text
images/heart_disease_distribution.png
```

---

#### Gender vs Heart Disease

Analyzes heart disease occurrence across genders.

**Output:**

```text
images/gender_vs_heart_disease.png
```

---

#### Chest Pain Type vs Heart Disease

Studies the relationship between chest pain categories and disease presence.

**Output:**

```text
images/chest_pain_vs_heart_disease.png
```

---

#### Correlation Heatmap

Displays relationships among all features.

**Output:**

```text
images/correlation_heatmap.png
```

---

# Step 4: Feature Selection

Features and target variable are separated.

### Features (X)

Medical attributes used for prediction.

### Target (y)

Heart disease status.

```text
0 → No Heart Disease
1 → Heart Disease
```

---

# Step 5: Train-Test Split

The dataset is divided into:

| Dataset | Percentage |
|----------|-----------|
| Training Data | 80% |
| Testing Data | 20% |

### Purpose

- Training data is used to learn patterns.
- Testing data is used to evaluate model performance.

---

# Step 6: Feature Scaling

StandardScaler is applied to normalize numerical features.

### Why?

Algorithms such as Logistic Regression perform better when features are on similar scales.

---

# Step 7: Model Training

Two classification algorithms were used.

---

## Logistic Regression

A widely used binary classification algorithm.

### Benefits

- Fast training
- Easy interpretation
- Suitable for medical prediction problems

---

## Decision Tree Classifier

Used for comparison with Logistic Regression.

### Benefits

- Easy visualization
- Captures nonlinear patterns
- Provides feature importance

---

# Step 8: Model Evaluation

The trained models are evaluated using multiple metrics.

---

## Accuracy Score

Measures overall prediction correctness.

### Formula

```text
Accuracy = Correct Predictions / Total Predictions
```

---

## Confusion Matrix

Shows:

- True Positives
- True Negatives
- False Positives
- False Negatives

**Output:**

```text
images/confusion_matrix.png
```

---

## ROC Curve

ROC (Receiver Operating Characteristic) Curve evaluates classification performance across different thresholds.

### Interpretation

Higher curve area indicates better model performance.

**Output:**

```text
images/roc_curve.png
```

---

## ROC-AUC Score

Measures the model's ability to distinguish between classes.

### Score Range

| AUC Score | Performance |
|------------|-------------|
| 0.50 | Random |
| 0.60 - 0.70 | Fair |
| 0.70 - 0.80 | Good |
| 0.80 - 0.90 | Very Good |
| 0.90+ | Excellent |

Screenshot available:

```text
screenshots/AUC_score.png
```

---

# Step 9: Feature Importance Analysis

Feature importance identifies which health factors most influence heart disease prediction.

### Importance Analysis Performed Using

- Logistic Regression Coefficients
- Decision Tree Feature Importance

### Output

```text
images/feature_importance.png
```

Screenshot:

```text
screenshots/features_importance.png
```

---

# Step 10: Model Saving

The final trained model is saved using Joblib.

### Saved Model

```text
outputs/models/heart_disease_model.pkl
```

### Why Save the Model?

The saved model can be reused later without retraining.

---

# 📈 Generated Visualizations

| Visualization | Location |
|---------------|----------|
| Heart Disease Distribution | images/heart_disease_distribution.png |
| Gender vs Heart Disease | images/gender_vs_heart_disease.png |
| Chest Pain Analysis | images/chest_pain_vs_heart_disease.png |
| Correlation Heatmap | images/correlation_heatmap.png |
| Confusion Matrix | images/confusion_matrix.png |
| ROC Curve | images/roc_curve.png |
| Feature Importance | images/feature_importance.png |

---

# 📸 Development Screenshots

The following screenshots document important project stages:

| Screenshot | Description |
|-------------|------------|
| loading_dataset_result.png | Dataset loading output |
| dataset_describe.png | Statistical summary |
| handling_missing_values.png | Missing value handling |
| available_and_final_column.png | Column validation |
| model_training_evaluation.png | Model training results |
| decision_tree_accuracy.png | Decision Tree performance |
| AUC_score.png | ROC-AUC score |
| features_importance.png | Feature importance output |
| model_saving.png | Model saving confirmation |

---

# 📊 Results Summary

### Completed Tasks

✅ Data Cleaning

✅ Missing Value Handling

✅ Exploratory Data Analysis

✅ Logistic Regression Model

✅ Decision Tree Model

✅ Accuracy Evaluation

✅ Confusion Matrix

✅ ROC Curve

✅ ROC-AUC Score

✅ Feature Importance Analysis

✅ Model Saving

---

# 🔮 Future Improvements

Potential enhancements include:

- Hyperparameter Tuning
- Cross Validation
- Random Forest Classifier
- XGBoost Classifier
- Model Deployment using Streamlit
- Real-time Heart Disease Prediction Application

---

# 🏁 Conclusion

This project successfully developed a machine learning pipeline for predicting heart disease risk using patient health data.

The workflow covered:

- Data preprocessing
- Exploratory analysis
- Classification modeling
- Performance evaluation
- Feature importance interpretation

The resulting model demonstrates how machine learning can assist in identifying individuals at risk of heart disease and support healthcare decision-making through data-driven insights.

---