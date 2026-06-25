# House Price Prediction

**DevelopersHub Corporation — AI/ML Internship | Task 6**

Predicts house sale prices using property features like living area, bedrooms, and overall quality rating. Built with Gradient Boosting Regressor and evaluated using MAE, RMSE, and R2 Score.

---

## Project Structure

```
task6-house-price-prediction/
│
├── data/
│   └── train.csv                        # Kaggle dataset (not pushed to GitHub)
│
├── images/
│   ├── actual_vs_predicted.png          # Model prediction accuracy chart
│   ├── area_vs_price.png                # Living area vs sale price scatter
│   ├── feature_importance.png           # Feature importance bar chart
│   └── price_distribution.png          # Sale price distribution histogram
│
├── screenshots/
│   ├── actual_vs_predicted_price_results.png
│   ├── additional_load_info.png
│   ├── data_preprocesing.png
│   ├── dataset_shape.png
│   ├── Exploratory Data Analysis.png
│   ├── feature_importance.png
│   ├── living_area_and_sale_price.png
│   ├── model_evaluation_results.png
│   └── summary.png
│
├── house_price_prediction.ipynb         # Full step by step notebook
├── house_price_prediction.py            # Standalone Python script
└── README.md                            # This file
```

---

## Task Overview

| Item | Detail |
|---|---|
| Task | Task 6 — House Price Prediction |
| Difficulty | Beginner |
| Model | Gradient Boosting Regressor |
| Dataset | House Prices — Advanced Regression Techniques (Kaggle) |
| Evaluation Metrics | MAE, RMSE, R2 Score |

---

## Dataset

**Source:** Kaggle — House Prices: Advanced Regression Techniques

**How to get it:**
1. Go to kaggle.com
2. Search "House Prices Advanced Regression Techniques"
3. Download and place `train.csv` inside the `data/` folder

The dataset has 1460 rows and 81 columns covering every measurable property of a house sold in Ames, Iowa.

---

## Features Used

| Feature | What it represents |
|---|---|
| GrLivArea | Above ground living area in square feet |
| BedroomAbvGr | Number of bedrooms above ground |
| FullBath | Number of full bathrooms |
| YearBuilt | Year the house was originally built |
| OverallQual | Overall material and finish quality (1 to 10) |
| GarageCars | Garage capacity in number of cars |
| TotalBsmtSF | Total basement area in square feet |

Target variable: **SalePrice** — the house sale price in USD.

---

## Setup and Installation

**Install all required libraries:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

**Option 1 — Notebook (recommended for step by step view):**

Open `house_price_prediction.ipynb` in VS Code and run cells from top to bottom.
Make sure to add `%matplotlib inline` at the top so charts render inside the notebook.

**Option 2 — Python script:**

```bash
python house_price_prediction.py
```

Both options produce the same 4 charts saved inside the `images/` folder.

---

## Process Walkthrough

**Step 1 — Load Data**

Load `train.csv` using pandas and preview the key columns like LotArea, Neighborhood, BedroomAbvGr, and SalePrice.

Expected output:
```
Dataset shape: (1460, 81)
```

**Step 2 — Preprocessing**

Select 7 relevant features, drop all other columns, and fill any missing values with the column median so no NaN values reach the model.

Expected output:
```
Missing values remaining: 0
Training rows: 1460
```

**Step 3 — Exploratory Data Analysis**

Two charts are generated here. The price distribution shows most houses sold between $100k and $250k with a right skew. The living area scatter shows a clear positive relationship — bigger houses cost more.

**Step 4 — Train the Model**

Split data 80/20 into train and test sets, then train a Gradient Boosting Regressor with 200 trees.

Expected output:
```
Training samples : 1168
Testing samples  : 292
Model trained!
```

**Step 5 — Evaluate**

Three metrics are calculated. A good result looks like:

```
Mean Absolute Error : $17,000
Root Mean Sq Error  : $26,000
R2 Score            : 0.91
```

R2 above 0.88 means the model explains over 88% of the price variation in unseen data.

**Step 6 — Actual vs Predicted Chart**

Teal dots plotted against a red dashed perfect prediction line. Dots clustered tightly around that line mean the model is predicting accurately.

**Step 7 — Feature Importance**

Horizontal bar chart showing which features the model relied on most. OverallQual and GrLivArea are consistently the top two, which matches real world real estate intuition.

---

## Results Summary

The Gradient Boosting model significantly outperforms plain Linear Regression on this dataset because it handles non-linear relationships between features and price. With just 7 features out of 81 available columns, it still achieves an R2 score above 0.90.

---

## Skills Demonstrated

- Regression modeling with Scikit-learn
- Feature selection and missing value handling
- Exploratory data analysis with Matplotlib and Seaborn
- Model evaluation using MAE, RMSE, and R2
- Real estate data understanding

---

## Author

**Muhammad Faizan**
AI/ML Engineering Intern — DevelopersHub Corporation
B.S. Computer Science — The Islamia University of Bahawalpur (2022–2026)

---

## Dataset Credit

Kaggle — House Prices: Advanced Regression Techniques
