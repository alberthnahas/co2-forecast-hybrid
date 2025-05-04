######################################################################################
# Purpose:      Predict and forecast ground-level CO₂ concentrations using a hybrid
#               Machine Learning (Random Forest) and Time Series (SARIMA) approach.
#
# Description:  This script loads monthly CO₂ and meteorological data from a CSV file.
#               It trains a Random Forest Regressor to predict ground-level CO₂ using
#               satellite-derived column concentrations (CO₂), CAMS column 
#               concentrations (CO, CH₄) and weather variables. The model's 
#               interpretability is enhanced with permutation and SHAP-based feature 
#               importance. The residual CO₂ time series is then modeled using SARIMA, 
#               with grid search based on AIC to determine optimal parameters. Forecasts 
#               with 95% confidence intervals are saved and visualized. A seasonal 
#               decomposition of predicted CO₂ is also included.
#
# Input File:   - A CSV file (e.g., "co2_downscaling_data.csv") containing columns for:
#                   'year', 'month', 'co2obs' (target), 'tcco2', 'tcco_1e4', 'tcch4_1e4',
#                   'u10', 'v10', 't2m', and 'mslp' (features).
#
# Output Files: - CSV: "co2_predictions_random_forest.csv" (Random Forest predictions)
#               - PNG: "rf_scatter_plot.png", "feature_importance_percent.png",
#                      "shap_bar_plot.png", "shap_summary_plot.png",
#                      "shap_dependence_dashboard.png",
#                      "co2_timeseries_comparison.png"
#               - CSV: "sarima_forecast_next_12_months.csv" (forecast + CI)
#               - PNG: "sarima_forecast_plot.png", "co2_last6_forecast12.png"
#               - CSV: "seasonal_decomposition_rf_co2.csv"
#               - PNG: "seasonal_decomposition_rf_co2.png",
#                      "trend_plus_seasonal_timeseries.png"
#
# Models Used:  - sklearn.ensemble.RandomForestRegressor
#               - statsmodels.tsa.statespace.SARIMAX
#               - shap.TreeExplainer
#
# Author:       Alberth Nahas (alberth.nahas@bmkg.go.id)
# Created Date: 2025-04-19
# Version:      1.2.0 (SHAP integrated, seasonal decomposition, full pipeline)
#
# Disclaimer:   This script is provided as-is for educational and operational use.
#               Ensure the input file matches expected column formats and monthly
#               temporal resolution. Model results are sensitive to data quality and
#               time coverage.
######################################################################################


import pandas as pd
import numpy as np
import shap
import os
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, KFold
import joblib
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# === Load dataset ===
df = pd.read_csv("co2_downscaling_data.csv")

# === Define features and target ===
X = df[["tcco2", "tcco_1e4", "tcch4_1e4", "u10", "v10", "t2m", "mslp"]]
y = df["co2obs"]

print("All libraries have been imported and the dataset has succesfully loaded")

# === Train default Random Forest model ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

# === Evaluate model ===
print(f"R = {np.corrcoef(y, y_pred)[0, 1]:.3f}")
print(f"R² = {r2_score(y, y_pred):.3f}")
print(f"RMSE = {np.sqrt(mean_squared_error(y, y_pred)):.3f} ppm")
print(f"MAE = {mean_absolute_error(y, y_pred):.3f} ppm")
print(f"MBE = {np.mean(y_pred - y):.3f} ppm")

# === Cross-validation ===
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
print(f"Cross-Validated R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# === Permutation importance ===
print("\n--- Permutation Importance Analysis ---")
importances = permutation_importance(model, X, y, n_repeats=30, random_state=42)
feature_importance = pd.Series(importances.importances_mean, index=X.columns).sort_values(ascending=False)

# Safely normalize and convert to percentages
total_importance = feature_importance.sum()

if total_importance > 0:
    feature_importance_percent = 100 * feature_importance / total_importance
else:
    feature_importance_percent = pd.Series(0, index=feature_importance.index)
    print("⚠️ Warning: Total importance is zero. Setting all values to 0%.")

# Sort for plotting
feature_importance_percent = feature_importance_percent.sort_values(ascending=True)
print("\nPermutation Importance in %:")
print(feature_importance_percent.to_string(float_format="{:.1f}".format))

# Plot
plt.figure(figsize=(8, 5))
feature_importance_percent.plot(kind='barh', color='steelblue')
plt.xlabel("Permutation Importance (%)")
plt.title("Feature Importance (Permutation-Based, % Scale)")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("feature_importance_percent.png", dpi=300)
plt.show()
print("✅ Saved: feature_importance_percent.png")

# === SHAP Analysis for Random Forest ===
print("\n--- SHAP Importance Analysis ---")

# Create SHAP TreeExplainer for Random Forest
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X)

# SHAP Bar Plot in Percentages
# Compute mean absolute SHAP values
mean_shap_values = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_shap_values, index=X.columns)

# Normalize to percentage
total = shap_importance.sum()
shap_importance_percent = 100 * shap_importance / total

# Sort values
shap_importance_percent = shap_importance_percent.sort_values(ascending=True)
print("\nSHAP Importance in %:")
print(shap_importance_percent.to_string(float_format="{:.1f}".format))

# Plot as horizontal bar chart
plt.figure(figsize=(8, 5))
shap_importance_percent.plot(kind="barh", color="dodgerblue")
plt.xlabel("SHAP Importance (%)")
plt.title("Feature Importance (SHAP, % Scale)")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("shap_bar_plot.png", dpi=300)
plt.show()
print("✅ Saved: shap_bar_plot_percent.png")

# Save SHAP summary (beeswarm)
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=300)
plt.show()
print("✅ Saved: shap_summary_plot.png")

# Ensure SHAP is using Matplotlib backend
shap.initjs()

# SHAP Dependence Dashboard
# Define features to include
features = X.columns.tolist()

# Create subplot grid
ncols = 3
nrows = int(np.ceil(len(features) / ncols))
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4.5*nrows))

# Flatten axes for easy indexing
axes = axes.flatten()

# Create each dependence plot using tcco2 for color
for i, feature in enumerate(features):
    #print(f"Plotting: {feature}")
    shap.dependence_plot(
        feature,
        shap_values,
        X,
        interaction_index="tcco2",  # Fixed color dimension
        ax=axes[i],
        show=False,
        dot_size=10,
        alpha=0.7
    )

# Remove any unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Add title and layout
fig.suptitle("SHAP Dependence Dashboard\nColor by: tcco2", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("shap_dependence_dashboard.png", dpi=300)
plt.show()
print("✅ Saved: shap_dependence_dashboard.png")

# === Save model and predictions ===
joblib.dump(model, "random_forest_co2_model.pkl")
df["predicted_rf_co2"] = y_pred
df.to_csv("co2_predictions_random_forest.csv", index=False)

# === Ensure datetime format ===
df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")

# === Plot time series ===
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["co2obs"], label="Observed CO₂ (co2obs)", color="red", linewidth=2)
plt.plot(df["date"], df["tcco2"], label="Total Column CO₂ (tcco2)", color="green", linestyle="--")
plt.plot(df["date"], df["predicted_rf_co2"], label="Predicted CO₂ (Random Forest)", color="blue")

plt.xlabel("Date")
plt.ylabel("CO₂ Concentration (ppm)")
plt.title("Time Series of Observed, Total Column, and Predicted CO₂")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("co2_timeseries_comparison.png", dpi=300)
plt.show()
print("✅ Saved: co2_timeseries_comparison.png")

# === Compute Metrics ===
def compute_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mbe = np.mean(y_pred - y_true)
    return r2, mae, mbe

# Metrics for tcco2
r2_tcco2, mae_tcco2, mbe_tcco2 = compute_metrics(df["co2obs"], df["tcco2"])

# Metrics for predicted_rf_co2
r2_pred, mae_pred, mbe_pred = compute_metrics(df["co2obs"], df["predicted_rf_co2"])

# === Print metrics ===
print("--- Metrics: co2obs vs tcco2 ---")
print(f"R² = {r2_tcco2:.3f}, MAE = {mae_tcco2:.3f} ppm, MBE = {mbe_tcco2:.3f} ppm")

print("\n--- Metrics: co2obs vs predicted_rf_co2 ---")
print(f"R² = {r2_pred:.3f}, MAE = {mae_pred:.3f} ppm, MBE = {mbe_pred:.3f} ppm")

# === Scatter plot: Observed vs Predicted ===
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, color='purple', alpha=0.6, label='Predicted vs Observed')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='1:1 Line')
plt.xlabel('Observed CO₂ (ppm)')
plt.ylabel('Predicted CO₂ (ppm)')
plt.title('Observed vs Predicted CO₂ (Random Forest)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_scatter_plot.png", dpi=300)
plt.show()
print("✅ Saved: rf_scatter_plot.png")

# === SARIMA Forecast ===
print("\n--- SARIMA Forecast for Next 12 Months ---")
df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
ts = pd.Series(df["predicted_rf_co2"].values, index=df["date"]).sort_index()

def optimize_sarima(ts, pdq_range, seasonal_pdq_range):
    best_aic = float("inf")
    best_order = None
    best_model = None
    for order in pdq_range:
        for s_order in seasonal_pdq_range:
            try:
                model = SARIMAX(ts, order=order, seasonal_order=s_order,
                                enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (order, s_order)
                    best_model = results
            except:
                continue
    return best_model, best_order

# Optimize SARIMA
p = d = q = range(0, 2)
pdq = [(x, y, z) for x in p for y in d for z in q]
seasonal_pdq = [(x, y, z, 12) for x in p for y in d for z in q]

sarima_result, best_order = optimize_sarima(ts, pdq, seasonal_pdq)
print(f"Best SARIMA order: {best_order}")

# Forecast
forecast = sarima_result.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

forecast_start = ts.index.max() + pd.DateOffset(months=1)
forecast_dates = pd.date_range(start=forecast_start, periods=12, freq='MS')

forecast_df = pd.DataFrame({
    "date": forecast_dates,
    "forecasted_co2": forecast_mean.values,
    "lower_ci": forecast_ci.iloc[:, 0].values,
    "upper_ci": forecast_ci.iloc[:, 1].values
})
forecast_df.to_csv("sarima_forecast_next_12_months.csv", index=False)
print(forecast_df)

# === Plot forecast ===
plt.figure(figsize=(10, 5))
plt.plot(ts.index, ts.values, label='Historical Predicted CO₂', color='gray')
plt.plot(forecast_df["date"], forecast_df["forecasted_co2"], label='SARIMA Forecast', color='blue')
plt.fill_between(forecast_df["date"], forecast_df["lower_ci"], forecast_df["upper_ci"], color='blue',
                 alpha=0.2, label="95% Confidence Interval")
plt.title("SARIMA Forecast of CO₂ (Next 12 Months)")
plt.xlabel("Date")
plt.ylabel("CO₂ (ppm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("sarima_forecast_plot.png", dpi=300)
plt.show()
print("✅ Saved: sarima_forecast_plot.png")

# === Filter last 6 months of historical data ===

# Load data
df_hist = pd.read_csv("co2_predictions_random_forest.csv")
df_forecast = pd.read_csv("sarima_forecast_next_12_months.csv")

# Ensure datetime format
df_hist["date"] = pd.to_datetime(df_hist["year"].astype(str) + "-" + df_hist["month"].astype(str).str.zfill(2) + "-01")
df_forecast["date"] = pd.to_datetime(df_forecast["date"])
last_6_months = df_hist.sort_values("date").iloc[-6:]

# Plot
plt.figure(figsize=(8, 8))
plt.plot(last_6_months["date"], last_6_months["predicted_rf_co2"], label="Last 6 Months (Predicted CO₂)", color="gray", linewidth=2)
plt.plot(df_forecast["date"], df_forecast["forecasted_co2"], label="Forecast (Next 12 Months)", color="blue")
plt.fill_between(df_forecast["date"], df_forecast["lower_ci"], df_forecast["upper_ci"],
                 color="blue", alpha=0.2, label="95% Confidence Interval")

plt.xlabel("Date")
plt.ylabel("CO₂ Concentration (ppm)")
plt.title("Predicted CO₂: Last 6 Months + 12-Month Forecast (SARIMA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("co2_last6_forecast12.png", dpi=300)
plt.show()
print("✅ Saved: co2_last6_forecast12.png")

# === Seasonal Decomposition of the predicted CO₂ ===
print("\n--- Seasonal Decomposition of Predicted CO₂ ---")
ts = ts.asfreq('MS')  # ensure monthly frequency for decomposition

decomposition = seasonal_decompose(ts, model='additive', period=12)

# Plot decomposition
fig = decomposition.plot()
fig.set_size_inches(10, 8)
plt.suptitle("Seasonal Decomposition of Predicted CO₂ (Additive)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("seasonal_decomposition_rf_co2.png", dpi=300)
plt.show()
print("✅ Saved: seasonal_decomposition_rf_co2.png")

# === Combine components into a DataFrame ===
decomp_df = pd.DataFrame({
    "date": ts.index,
    "predicted": ts.values,
    "trend": decomposition.trend,
    "seasonal": decomposition.seasonal,
    "residual": decomposition.resid,
    "trend_plus_seasonal": decomposition.trend + decomposition.seasonal
})

# === Save decomposition result to CSV ===
decomp_df.to_csv("seasonal_decomposition_rf_co2.csv", index=False)
print("Saved: seasonal_decomposition_rf_co2.csv")

# === Plot trend + seasonal overlay ===
plt.figure(figsize=(10, 5))
plt.plot(decomp_df["date"], decomp_df["predicted"], label="Predicted CO₂", color='gray', alpha=0.6)
plt.plot(decomp_df["date"], decomp_df["trend_plus_seasonal"], label="Trend + Seasonal", color='blue')
plt.title("Predicted CO₂ vs Trend + Seasonal Component")
plt.xlabel("Date")
plt.ylabel("CO₂ (ppm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("trend_plus_seasonal_timeseries.png", dpi=300)
plt.show()
print("✅ Saved: trend_plus_seasonal_timeseries.png")

# === Optional for Google Drive ===
#from google.colab import drive
#drive.mount('/content/drive')
#%cd /content/drive/MyDrive/ # Write the directory on Google Drive you want to work from
