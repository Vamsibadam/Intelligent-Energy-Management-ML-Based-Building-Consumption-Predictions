import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import math
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from statsmodels.tsa.arima.model import ARIMA


# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("electricity_bill_dataset.csv")  # Replace with your file
    return df

df = load_data()

# Sidebar - Filters
st.sidebar.header("Filters")

# Set default values to the first available option
selected_city = st.sidebar.multiselect("Select City", df["City"].unique(), default=[df["City"].unique()[0]])
selected_month = st.sidebar.multiselect("Select Month", df["Month"].unique(), default=[df["Month"].unique()[0]])
selected_company = st.sidebar.multiselect("Select Company", df["Company"].unique(), default=[df["Company"].unique()[0]])

# Filter Data
filtered_df = df[
    (df["City"].isin(selected_city)) & 
    (df["Month"].isin(selected_month)) & 
    (df["Company"].isin(selected_company))
]

st.title("🏠 Indian Household Electricity Consumption Analysis")
st.write("This app provides interactive insights into energy consumption in Indian households.")

# Display Dataset
st.subheader("📊 Filtered Data Preview")
st.dataframe(filtered_df)

# Appliance Selection (Checkboxes)
st.sidebar.subheader("Appliance Selection")
appliances = ["Fan", "Refrigerator", "AirConditioner", "Television", "Monitor", "MotorPump"]
selected_appliances = {appliance: st.sidebar.checkbox(appliance, value=(appliance == "Fan")) for appliance in appliances}




# Graphs for Appliance Usage
st.subheader("📉 Appliance Usage Analysis")
fig, ax = plt.subplots()
for appliance, selected in selected_appliances.items():
    if selected:
        sns.lineplot(data=filtered_df, x="Month", y=appliance, label=appliance, ax=ax)
plt.xlabel("Month")
plt.ylabel("Usage (Hours)")
plt.title("Appliance Usage Over Time")
st.pyplot(fig)

# Energy Bill Distribution
st.subheader("⚡ Electricity Bill Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(filtered_df["ElectricityBill"], bins=20, kde=True, color="blue", ax=ax)
plt.xlabel("Electricity Bill (INR)")
plt.ylabel("Frequency")
st.pyplot(fig)

# ML Model - Predicting Energy Consumption
st.subheader("📈 Energy Consumption Prediction")

# Select Features & Target
features = ["Fan", "Refrigerator", "AirConditioner", "Television", "Monitor", "MotorPump", "MonthlyHours", "TariffRate"]
X = filtered_df[features]
y = filtered_df["ElectricityBill"]

# Handle missing values
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN Model
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Calculate RMSE & MAPE
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

st.write(f"✅ **Root Mean Square Error (RMSE):** {rmse:.2f}")
st.write(f"✅ **Mean Absolute Percentage Error (MAPE):** {mape:.2f}%")

# Predictive Alerts - Detect abnormal consumption
# Calculate per-unit cost based on the dataset
df["CostPerUnit"] = df["ElectricityBill"] / (df["MonthlyHours"] * df["TariffRate"])

# Define threshold for each tariff rate
tariff_threshold = df.groupby("TariffRate")["CostPerUnit"].mean() + (1.5 * df.groupby("TariffRate")["CostPerUnit"].std())

# Predict electricity bill using trained model
y_pred = model.predict(X_test_scaled)

# Calculate RMSE & MAPE
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

# Get the corresponding tariff rate from the test set
test_tariff_rate = X_test["TariffRate"].values

# Compute predicted cost per unit
predicted_cost_per_unit = y_pred / (X_test["MonthlyHours"].values * test_tariff_rate)

# Fit ARIMA Model
model = ARIMA(df["ElectricityBill"], order=(5,1,0))  
model_fit = model.fit()
forecast = model_fit.forecast(steps=3)  # Predict next 3 months

st.subheader("🔮 Future Electricity Bill Prediction")
st.write(f"📅 Next Month's Predicted Bill: ₹{forecast.iloc[0]:.2f}")

# Check if the predicted cost per unit exceeds the threshold for the given tariff rate
for i, cost in enumerate(predicted_cost_per_unit):
    applicable_threshold = tariff_threshold.get(test_tariff_rate[i], None)
    if applicable_threshold and cost > applicable_threshold:
        st.error(f"⚠️ **Alert! High Electricity Consumption Detected for Tariff Rate {test_tariff_rate[i]:.2f}!**")
        break  # Show alert only once



# Dynamic Tariff Estimation
st.subheader("📊 Dynamic Tariff Estimation")
tariff_rate = st.slider("Select Current Tariff Rate (INR per unit)", min_value=3.0, max_value=10.0, value=filtered_df["TariffRate"].mean())
estimated_bill = tariff_rate * filtered_df["MonthlyHours"].mean()

st.write(f"💰 **Estimated Monthly Bill:** ₹{estimated_bill:.2f}")

# Energy Saving Recommendations
st.subheader("💡 Energy Saving Recommendations")
if mape > 20:
    st.warning("High prediction error! Consider checking appliance efficiency and optimizing energy use.")
st.write("🔹 Use LED bulbs instead of CFLs to reduce electricity consumption.")
st.write("🔹 Unplug unused devices to prevent phantom power loss.")
st.write("🔹 Optimize air conditioning by keeping temperature around 24°C.")

# Generate PDF Report
def generate_pdf():
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)
    pdf.drawString(100, 750, "Indian Household Electricity Consumption Report")
    pdf.drawString(100, 730, f"Cities Selected: {', '.join(selected_city)}")
    pdf.drawString(100, 710, f"Companies Selected: {', '.join(selected_company)}")
    pdf.drawString(100, 690, f"RMSE: {rmse:.2f}")
    pdf.drawString(100, 670, f"MAPE: {mape:.2f}%")
    pdf.drawString(100, 650, f"Estimated Bill: ₹{estimated_bill:.2f}")
    pdf.save()
    buffer.seek(0)
    return buffer

if st.button("📥 Download Report"):
    pdf_data = generate_pdf()
    st.download_button(label="Download PDF", data=pdf_data, file_name="energy_report.pdf", mime="application/pdf")

st.success("🎯 Interactive Energy Management System Ready!")
