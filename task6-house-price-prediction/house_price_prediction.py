import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('./data/train.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df[['LotArea', 'BedroomAbvGr', 'Neighborhood', 'SalePrice']].head(10))

features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'OverallQual', 'GarageCars', 'TotalBsmtSF']
target = 'SalePrice'

df_model = df[features + [target]].copy()
df_model.fillna(df_model.median(), inplace=True)
print(f"Missing values remaining: {df_model.isnull().sum().sum()}")

plt.figure(figsize=(8, 4))
sns.histplot(df_model['SalePrice'], bins=40, kde=True, color='coral')
plt.title('Distribution of House Sale Prices')
plt.xlabel('Sale Price (USD)')
plt.tight_layout()
plt.savefig('./images/price_distribution.png', dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(df_model['GrLivArea'], df_model['SalePrice'], alpha=0.4, color='steelblue')
plt.xlabel('Above Ground Living Area (sq ft)')
plt.ylabel('Sale Price (USD)')
plt.title('Living Area vs Sale Price')
plt.tight_layout()
plt.savefig('./images/area_vs_price.png', dpi=150)
plt.show()

X = df_model[features]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model trained!")

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error : ${mae:,.0f}")
print(f"Root Mean Sq Error  : ${rmse:,.0f}")
print(f"R2 Score            : {r2:.4f}")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price (USD)')
plt.ylabel('Predicted Price (USD)')
plt.title('Actual vs Predicted House Prices')
plt.legend()
plt.tight_layout()
plt.savefig('./images/actual_vs_predicted.png', dpi=150)
plt.show()

importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True)

plt.figure(figsize=(7, 5))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='steelblue')
plt.title('Feature Importance — House Price Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('./images/feature_importance.png', dpi=150)
plt.show()

print("\nAll charts saved. Task 6 complete.")
