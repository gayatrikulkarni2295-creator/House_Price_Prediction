import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 50)
print("   HOUSE PRICE PREDICTION - ML MODEL")
print("=" * 50)

print("\n Loading dataset...")
df = pd.read_csv('house_data.csv')
print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"\n   Sample data:\n{df.head(3).to_string()}")
print("\n Basic Statistics:")
print(f"   Average Price : ₹{df['price'].mean():,.0f}")
print(f"   Min Price     : ₹{df['price'].min():,.0f}")
print(f"   Max Price     : ₹{df['price'].max():,.0f}")
print(f"   Avg Area      : {df['area_sqft'].mean():.0f} sqft")

# ── 3. Feature Engineering ────────────────────────────
print("\n Preparing features...")
X = df[['area_sqft', 'bedrooms', 'bathrooms', 'house_age_years', 'garage', 'location_score']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training samples : {len(X_train)}")
print(f"   Testing samples  : {len(X_test)}")

print("\n Training Models...")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
print(f"   Linear Regression R² Score : {lr_r2:.4f}")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
print(f"   Random Forest R² Score     : {rf_r2:.4f}")
print("\n Best Model Evaluation (Random Forest):")
mae  = mean_absolute_error(y_test, rf_pred)
rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
r2   = r2_score(y_test, rf_pred)

print(f"   MAE  (Mean Absolute Error) : ₹{mae:,.0f}")
print(f"   RMSE (Root Mean Sq Error)  : ₹{rmse:,.0f}")
print(f"   R²   (Accuracy Score)      : {r2:.4f} ({r2*100:.1f}%)")
print("\n Feature Importance:")
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
for feat, imp in importances.items():
    bar = '█' * int(imp * 40)
    print(f"   {feat:<20} {bar} {imp:.3f}")
print("\n🏠 Sample Predictions:")
sample = pd.DataFrame({
    'area_sqft'       : [1200, 2500, 3500],
    'bedrooms'        : [2,    4,    5   ],
    'bathrooms'       : [1,    2,    3   ],
    'house_age_years' : [10,   5,    1   ],
    'garage'          : [0,    1,    1   ],
    'location_score'  : [5,    7,    9   ]
})
preds = rf.predict(sample)
for i, (_, row) in enumerate(sample.iterrows()):
    print(f"   House {i+1}: {int(row['area_sqft'])} sqft, "
          f"{int(row['bedrooms'])} bed, score={int(row['location_score'])} "
          f"→ Predicted: ₹{preds[i]:,.0f}")
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("\n✅ Model saved as house_price_model.pkl")
print("=" * 50)
