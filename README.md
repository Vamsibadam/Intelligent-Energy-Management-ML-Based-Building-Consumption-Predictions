# 🔌 Indian Household Electricity Consumption Analysis

This project provides a comprehensive and interactive dashboard built with **Streamlit** for analyzing and predicting household electricity usage based on appliance-level data. It includes predictive modeling, anomaly detection, and dynamic tariff estimation.

---

## 📁 Project Structure

```
├── electricity_bill_dataset.csv     # Dataset used for analysis
├── energy_analysis_app.py           # Main Streamlit app
├── README.md                        # This documentation
└── requirements.txt                 # Python dependencies
```

---

## 🚀 Features

* 📊 Filter energy data by city, month, and electricity company
* 📉 Visualize appliance usage trends and electricity bill distribution
* 🤖 Predict electricity bills using **K-Nearest Neighbors**
* 🔮 Forecast future bills using **ARIMA time series model**
* ⚠️ Detect abnormal electricity consumption based on dynamic thresholds
* 💡 Energy-saving recommendations based on model performance
* 📄 Generate downloadable PDF reports summarizing the analysis

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Vamsibadam/Intelligent-Energy-Management-ML-Based-Building-Consumption-Predictions
cd Intelligent-Energy-Management-ML-Based-Building-Consumption-Predictions
```

### 2. Install dependencies

Make sure Python 3.8+ is installed, then:

```bash
pip install -r requirements.txt
```

---

## 🧪 Running the App

```bash
streamlit run energy_analysis_app.py
```

Open the link provided by Streamlit in your browser (usually [http://localhost:8501](http://localhost:8501)).

---

## 📈 Technologies Used

* **Streamlit** – Front-end dashboard
* **scikit-learn** – KNN Regression model
* **statsmodels** – ARIMA time series forecasting
* **matplotlib & seaborn** – Data visualization
* **pandas & numpy** – Data processing
* **reportlab** – PDF report generation

---

## 📂 Dataset

Ensure that `electricity_bill_dataset.csv` is in the root folder. The dataset should include:

* `City`, `Month`, `Company`
* Appliance usage: `Fan`, `Refrigerator`, `AirConditioner`, `Television`, `Monitor`, `MotorPump`
* `MonthlyHours`, `TariffRate`, `ElectricityBill`

---

## 📥 Export PDF Report

Click on **"📥 Download Report"** to generate a summary PDF with:

* Selected cities and companies
* RMSE & MAPE from predictions
* Estimated bill
* Insights

---

## 📌 Future Improvements

* Incorporate more ML models (e.g., Random Forest, XGBoost)
* Real-time data integration via APIs
* Multi-user support with authentication

---

## 🤝 Contribution

Feel free to open issues or submit pull requests to improve functionality, performance, or visual design.

---



