# 📈 Stock Price Prediction Using Linear Regression

## 🚀 Project Overview

This project is a **Machine Learning-based Stock Price Prediction System** that uses **Linear Regression** to estimate the future closing price of selected NSE-listed companies.

The application provides a simple interface where the user can:

- 🏢 Select a company
- 📅 Select a future date
- 🔮 Predict the estimated stock closing price
- 📊 View the historical price and prediction on a graph

The project is designed as a **Machine Learning practical project** to demonstrate the application of Linear Regression on historical stock-market data.

---

## 🎯 Objective

The main objective of this project is to demonstrate how **Linear Regression** can be used to predict a continuous numerical value such as a stock's closing price using historical stock-market data.

The system takes historical stock data, identifies the relationship between trading days and closing prices, trains a Linear Regression model, and uses the trained model to estimate a future stock price.

---

## 💡 Problem Statement

Stock prices change continuously and are influenced by various market factors.

The objective of this project is to use historical stock-price data and a Machine Learning algorithm to estimate the future closing price of a selected stock.

The user can select a company and a future date, after which the system generates an estimated stock price using Linear Regression.

---

## 🏢 Companies Included

The application supports multiple NSE-listed companies:

| Company | NSE Symbol |
|---|---|
| 🏦 State Bank of India | SBIN |
| 🏭 Reliance Industries | RELIANCE |
| 💻 Tata Consultancy Services | TCS |
| 💻 Infosys | INFY |
| 🏦 HDFC Bank | HDFCBANK |
| 🏦 ICICI Bank | ICICIBANK |
| 🏢 ITC | ITC |
| 🏗️ Larsen & Toubro | LT |
| 📱 Bharti Airtel | BHARTIARTL |
| 🚗 Tata Motors | TATAMOTORS |

---

## 🤖 Machine Learning Algorithm

### Linear Regression

The project uses **Linear Regression**, a supervised Machine Learning algorithm used for predicting continuous numerical values.

In this project:

**Independent Variable (X):**
> Trading Day Number

**Dependent Variable (Y):**
> Closing Stock Price

The basic Linear Regression equation is:

```text
y = mx + c

Where:

y = Predicted Stock Closing Price
x = Future Trading Day Number
m = Slope of the regression line
c = Intercept

The model learns the best-fit relationship between historical trading days and their corresponding closing prices.

📊 How the Prediction Works

The complete prediction process is:

Historical NSE Stock Data
          ↓
     Data Cleaning
          ↓
   Trading Day Number
          ↓
 Closing Stock Price
          ↓
 Train Linear Regression
          ↓
    User Selects Company
          ↓
    User Selects Future Date
          ↓
 Check Valid Trading Day
          ↓
Calculate Future Trading Day
          ↓
   Linear Regression Model
          ↓
  Predicted Closing Price
          ↓
    Display Result + Graph
Step-by-step process
📥 Historical NSE stock data is loaded.
🧹 The data is cleaned and arranged according to trading dates.
🔢 Each trading date is represented using a trading-day number.
📈 The historical closing price is used as the target variable.
🤖 A separate Linear Regression model is used for each company.
🏢 The user selects a company from the dropdown.
📅 The user selects a future date.
🗓️ The system checks whether the selected date is a valid NSE trading day.
🔢 The future trading-day number is calculated.
🔮 The trained Linear Regression model predicts the estimated closing price.
📊 The predicted value is displayed along with a graph.
⏱️ Dynamic Prediction

The application allows the user to enter different future dates rather than restricting the prediction to only 10 days.

For example, the user can select:

📅 15 April 2026
📅 30 April 2026
📅 15 May 2026
📅 1 June 2026
📅 1 July 2026

The system dynamically calculates the corresponding future trading-day number and generates the prediction using the trained model.

📅 Trading Day Validation

The application checks whether the selected date is a valid trading day.

It does not consider:

Saturday
Sunday
Applicable NSE market holidays

If the selected date is not a trading day, the system asks the user to select another date.

Example:

⚠️ Selected date is not an NSE trading day.
Please select another date.

📂 Dataset

The project uses historical stock-market data for the financial year:

1 April 2025 to 31 March 2026

The datasets contain historical information such as:

📅 Date
🔓 Open
📈 High
📉 Low
💰 Close
📊 Volume

The Close value is used as the target variable for prediction.

Each company has its own historical dataset.

📈 Graph Visualization

The application displays a graph containing:

Historical closing prices
Linear Regression trend line
Future predicted price

The graph dynamically changes according to the selected company and future date.

X-axis: Date

Y-axis: Closing Price (₹)

🛠️ Technologies Used
Programming Language
🐍 Python
Machine Learning
🤖 Scikit-learn
📐 Linear Regression
Data Processing
🐼 Pandas
Backend
⚡ FastAPI
🚀 Uvicorn
Frontend
🌐 HTML
🎨 CSS
⚙️ JavaScript
Visualization
📊 Plotly
Data Source
📈 NSE historical stock data
✨ Key Features
🏢 Multiple company selection
📅 Future date selection
🔮 Future stock-price prediction
🤖 Linear Regression Machine Learning model
📊 Interactive stock-price graph
🗓️ Trading-day validation
📈 Historical price visualization
💻 Simple localhost application
🔄 Dynamic prediction based on user input
🖥️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/shweeta1015/stock-price-prediction-linear-regression.git
2️⃣ Open the Project Folder
cd stock-price-prediction-linear-regression
3️⃣ Install Required Libraries
pip install pandas scikit-learn fastapi uvicorn plotly
4️⃣ Run the Application
python app.py
5️⃣ Open the Localhost Application

Open your browser and visit:

http://127.0.0.1:5000
🔄 Example Workflow
🏢 Select Company
       ↓
📅 Select Future Date
       ↓
🗓️ Check Trading Day
       ↓
🔢 Calculate Future Trading Day
       ↓
🤖 Apply Linear Regression
       ↓
🔮 Generate Prediction
       ↓
💰 Display Predicted Price
       ↓
📊 Display Graph
🧠 Why Linear Regression?

Linear Regression is suitable for this project because the output we want to predict is a continuous numerical value, which is the stock's closing price.

For example:

₹725.40
₹731.25
₹745.80

Unlike classification algorithms that predict categories, Linear Regression predicts numerical values.

🔍 Linear Regression vs Logistic Regression
Linear Regression	Logistic Regression
Predicts continuous values	Mainly used for classification
Example: Stock Price	Example: Buy/Sell
Output can be ₹750.50	Output is a class/probability
Used in this project	Not used in this project

Therefore, Linear Regression is used because the required output is a stock price.

⚠️ Important Disclaimer

The stock prices generated by this application are estimated predictions produced by a Linear Regression model based on historical data.

Actual stock prices can be affected by many factors such as:

Market conditions
Company performance
Economic events
News
Investor sentiment
Government policies
Global events

Therefore, the predicted value may differ significantly from the actual future market price.

This project is developed for educational and academic purposes only and should not be considered financial advice or a recommendation to buy or sell stocks.

🎓 Academic Purpose

This project demonstrates the practical implementation of:

📚 Machine Learning
📊 Data Analysis
📈 Linear Regression
🐍 Python Programming
🌐 Web Application Development
📉 Data Visualization

It demonstrates how historical real-world data can be used to build a simple Machine Learning prediction system.

👩‍💻 Author

Shweta Thorat

B.Tech Computer Science & Engineering

🎓 Machine Learning Project
📈 Stock Price Prediction using Linear Regression

⭐ If you find this project useful, feel free to explore the repository!
