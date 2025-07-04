import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv("./monthly_outbound_summary_filled.csv")
df.rename(columns={'MONTH': 'YEAR_MONTH'}, inplace=True)
df.rename(columns={'TOTAL_QUANTITY': 'TOTAL_OUTBOUND_MT'}, inplace=True)

df["YEAR_MONTH"] = pd.to_datetime(df["YEAR_MONTH"], format="%Y-%m")

df = df.sort_values(by=["PLANT_NAME", "YEAR_MONTH"]).reset_index(drop=True)

df["OUTBOUND_LAG_1"] = df.groupby("PLANT_NAME")["TOTAL_OUTBOUND_MT"].shift(1)
df["OUTBOUND_LAG_2"] = df.groupby("PLANT_NAME")["TOTAL_OUTBOUND_MT"].shift(2)
df["OUTBOUND_ROLLING_MEAN_3"] = (
    df.groupby("PLANT_NAME")["TOTAL_OUTBOUND_MT"].shift(1).rolling(3).mean()
)

df["MONTH"] = df["YEAR_MONTH"].dt.month
df["YEAR"] = df["YEAR_MONTH"].dt.year
df["MONTH_SIN"] = np.sin(2 * np.pi * df["MONTH"] / 12)
df["MONTH_COS"] = np.cos(2 * np.pi * df["MONTH"] / 12)

df.dropna(
    subset=["OUTBOUND_LAG_1", "OUTBOUND_LAG_2", "OUTBOUND_ROLLING_MEAN_3"],
    inplace=True,
)

df["PLANT_NAME_ORIGINAL"] = df["PLANT_NAME"]
df["MATERIAL_NAME_ORIGINAL"] = df["MATERIAL_NAME"]

# One-hot encode PLANT_NAME and MATERIAL_NAME
df = pd.get_dummies(df, columns=["PLANT_NAME", "MATERIAL_NAME"], drop_first=True)

df["LABEL"] = (
    df["YEAR_MONTH"].astype(str)
    + " | "
    + df["PLANT_NAME_ORIGINAL"]
    + " | "
    + df["MATERIAL_NAME_ORIGINAL"]
)

target = "TOTAL_OUTBOUND_MT"
excluded = [
    "YEAR_MONTH",
    "TOTAL_OUTBOUND_MT",
    "MONTH",
    "YEAR",
    "LABEL",
    "PLANT_NAME_ORIGINAL",
    "MATERIAL_NAME_ORIGINAL",
]
features = [col for col in df.columns if col not in excluded]

X = df[features]
y = df[target]

labels_test = df.loc[X.index, "LABEL"]

# Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# --- Predict ---
y_pred = model.predict(X)



# Result reporting
result_df = df.loc[X.index, ["YEAR_MONTH", "PLANT_NAME_ORIGINAL", "MATERIAL_NAME_ORIGINAL"]].copy()
result_df["ACTUAL_OUTBOUND_MT"] = y.values
result_df["PREDICTED_OUTBOUND_MT"] = y_pred
result_df["LABEL"] = df["LABEL"]

result_df.to_csv("./outbound_predictions_for_powerbi.csv")
