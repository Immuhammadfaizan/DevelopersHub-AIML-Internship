# Iris Dataset Visualization and Exploratory Data Analysis (EDA)

## Project Overview

This project demonstrates the process of loading, exploring, analyzing, and visualizing the famous **Iris Dataset** using Python. The goal is to understand the dataset through descriptive statistics and visualizations while practicing fundamental data analysis techniques with **Pandas**, **Matplotlib**, and **Seaborn** in a Jupyter Notebook environment.

---

## Objective

The main objectives of this task are:

- Load and inspect a dataset using Pandas.
- Explore dataset structure and summary statistics.
- Understand feature distributions.
- Analyze relationships between variables.
- Detect potential outliers.
- Create visualizations using both Matplotlib and Seaborn.

---

## Dataset Information

The Iris dataset contains measurements of iris flowers from three different species:

- Setosa
- Versicolor
- Virginica

### Features

| Feature | Description |
|----------|-------------|
| sepal length (cm) | Length of sepal |
| sepal width (cm) | Width of sepal |
| petal length (cm) | Length of petal |
| petal width (cm) | Width of petal |
| species | Flower species |

### Dataset Size

- Rows: 150
- Columns: 5

---

# Project Structure

```text
task1-dataset-visualization/
│
├── data/
│   └── iris.csv
│
├── images/
│   ├── correlation_matrix.png
│   ├── histogram_of_features.png
│   ├── outliers_using_boxing_plots.png
│   ├── pair_plot.png
│   ├── petal_length_box_plot.png
│   ├── petal_length_vs_petal_width.png
│   └── seaborn_histogram_sepal_length.png
│
├── data.py
├── Iris_Analysis.ipynb
└── README.md
```

---

# Technologies Used

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# Installation and Setup

## 1. Clone the Repository

```bash
git clone <repository-url>
cd task1-dataset-visualization
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Iris_Analysis.ipynb
```

---

# Task Implementation

## Step 1: Load Dataset

The Iris dataset was loaded from:

```text
data/iris.csv
```

Using Pandas:

```python
import pandas as pd

df = pd.read_csv("data/iris.csv")
```

---

## Step 2: Inspect Dataset

Basic dataset inspection was performed using:

### Dataset Shape

```python
df.shape
```

### Column Names

```python
df.columns
```

### First Five Rows

```python
df.head()
```

---

## Step 3: Dataset Information

Information about:

- Data types
- Non-null values
- Memory usage

was obtained using:

```python
df.info()
```

---

## Step 4: Descriptive Statistics

Summary statistics including:

- Mean
- Median
- Standard Deviation
- Minimum
- Maximum
- Quartiles

were generated using:

```python
df.describe()
```

---

# Data Visualization

The assignment required the use of both **Matplotlib** and **Seaborn** for visualization.

---

## 1. Scatter Plot

### File

```text
images/petal_length_vs_petal_width.png
```

### Purpose

Shows the relationship between:

- Petal Length
- Petal Width

Useful for identifying patterns and species separation.

---

## 2. Histogram of Features

### File

```text
images/histogram_of_features.png
```

### Purpose

Displays the distribution of all numerical features and helps understand:

- Spread of values
- Skewness
- Frequency distribution

---

## 3. Seaborn Histogram

### File

```text
images/seaborn_histogram_sepal_length.png
```

### Purpose

Visualizes the distribution of Sepal Length using Seaborn with a density curve.

---

## 4. Box Plot for Outlier Detection

### File

```text
images/outliers_using_boxing_plots.png
```

### Purpose

Used to identify:

- Outliers
- Data spread
- Quartile ranges

---

## 5. Petal Length Box Plot by Species

### File

```text
images/petal_length_box_plot.png
```

### Purpose

Compares petal length across different Iris species.

---

## 6. Pair Plot

### File

```text
images/pair_plot.png
```

### Purpose

Provides pairwise comparisons between all numerical features.

Helps identify:

- Correlations
- Clusters
- Species separation

---

## 7. Correlation Matrix

### File

```text
images/correlation_matrix.png
```

### Purpose

Displays correlation coefficients between numerical variables.

Helps understand:

- Strong positive correlations
- Weak correlations
- Feature relationships

---

# Key Findings

### Dataset Quality

- No missing values were found.
- Dataset is clean and ready for analysis.

### Feature Relationships

- Petal Length and Petal Width are highly correlated.
- Sepal features show weaker relationships compared to petal features.

### Species Separation

- Setosa is clearly separated from the other species.
- Versicolor and Virginica show some overlap.

### Outlier Analysis

- A few potential outliers are visible in sepal measurements.
- No significant anomalies affecting analysis.

---

# Libraries Used

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
```

---

# Skills Demonstrated

- Data Loading
- Data Cleaning and Inspection
- Exploratory Data Analysis (EDA)
- Descriptive Statistics
- Data Visualization
- Correlation Analysis
- Outlier Detection
- Python Programming
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Assignment Requirements Checklist

| Requirement | Status |
|------------|---------|
| Load dataset using Pandas | ✅ Completed |
| Print shape of dataset | ✅ Completed |
| Print column names | ✅ Completed |
| Display first rows using head() | ✅ Completed |
| Use info() | ✅ Completed |
| Use describe() | ✅ Completed |
| Create scatter plot | ✅ Completed |
| Create histograms | ✅ Completed |
| Create box plots | ✅ Completed |
| Use Matplotlib | ✅ Completed |
| Use Seaborn | ✅ Completed |

---

# Conclusion

This project successfully demonstrates the complete workflow of Exploratory Data Analysis (EDA) using the Iris Dataset. By leveraging Pandas for data manipulation and Matplotlib/Seaborn for visualization, valuable insights were extracted regarding feature distributions, correlations, species separation, and potential outliers. The project serves as a strong foundation for further studies in data analytics, machine learning, and data visualization.