# 📈 Stock Price Prediction Using Linear Regression

## 🚀 Project Overview

This project is a **Machine Learning-based Stock Price Prediction System** that uses **Linear Regression** to estimate the future closing price of selected NSE-listed companies.

The application allows the user to:

- 🏢 Select a company
- 📅 Select a future date
- 🔮 Predict the estimated closing price
- 📊 View historical prices and the prediction on a graph

The project is developed as an **academic Machine Learning practical project** to demonstrate the use of Linear Regression with historical stock-market data.



## 🎯 Objective

The main objective of this project is to demonstrate how **Linear Regression** can be used to predict a continuous numerical value such as a stock's closing price.

The system uses historical stock data, represents trading dates using trading-day numbers, trains a Linear Regression model, and uses the trained model to estimate a future closing price.



## 💡 Problem Statement

Stock prices change continuously and are influenced by several market factors.

This project aims to use historical stock-price data and a Machine Learning algorithm to estimate the future closing price of a selected stock.

The user can select a company and a future date, and the system generates an estimated closing price using Linear Regression.



## 🏢 Companies Included

The application supports the following NSE-listed companies:

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



## 🤖 Machine Learning Algorithm

### Linear Regression

The project uses **Linear Regression**, a supervised Machine Learning algorithm used to predict continuous numerical values.

### Variables Used

**Independent Variable (X):**
> Trading Day Number

**Dependent Variable (Y):**
> Closing Stock Price

### Linear Regression Equation

**y = mx + c**

Where:

- `y` = Predicted Stock Closing Price
- `x` = Trading Day Number
- `m` = Slope of the regression line
- `c` = Intercept

The model learns the best-fit relationship between historical trading days and their corresponding closing prices.



## 📊 How the Prediction Works

The overall prediction process is:

Historical NSE Stock Data  
↓  
Data Cleaning  
↓  
Trading Day Number  
↓  
Closing Stock Price  
↓  
Train Linear Regression Model  
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

### Step-by-Step Process

1. 📥 Historical NSE stock data is loaded.
2. 🧹 The data is cleaned and arranged according to trading dates.
3. 🔢 Each trading date is represented using a trading-day number.
4. 📈 The historical closing price is used as the target variable.
5. 🤖 A separate Linear Regression model is used for each company.
6. 🏢 The user selects a company from the application.
7. 📅 The user selects a future date.
8. 🗓️ The system checks whether the selected date is a valid trading day.
9. 🔢 The corresponding future trading-day number is calculated.
10. 🔮 The trained Linear Regression model predicts the estimated closing price.
11. 📊 The predicted value is displayed along with a graph.



## ⏱️ Dynamic Prediction

The application allows the user to select different future dates instead of restricting the prediction to a fixed number of days.

For example, the user can select:

- 📅 15 April 2026
- 📅 30 April 2026
- 📅 15 May 2026
- 📅 1 June 2026
- 📅 1 July 2026

The system dynamically calculates the corresponding trading-day number and generates the prediction using the trained Linear Regression model.



## 📅 Trading Day Validation

The application checks whether the selected date is a valid trading day.

The system does not consider:

- Saturday
- Sunday
- Applicable NSE market holidays

If the selected date is not a valid trading day, the system asks the user to select another date.

### Example

> ⚠️ Selected date is not an NSE trading day.
>
> Please select another date.



## 📂 Dataset

The project uses historical stock-market data for the financial year:

**1 April 2025 to 31 March 2026**

The datasets contain information such as:

- 📅 Date
- 🔓 Open
- 📈 High
- 📉 Low
- 💰 Close
- 📊 Volume

The **Close** value is used as the target variable for prediction.

Each supported company has its own historical dataset.



## 📈 Graph Visualization

The application provides graphical visualization of the stock data.

The graph displays:

- 📈 Historical closing prices
- 📉 Linear Regression trend line
- 🔮 Future predicted price

The graph dynamically changes according to the selected company and prediction date.

**X-axis:** Date

**Y-axis:** Closing Price (₹)



## 🛠️ Technologies Used

| Category | Technology |
|---|---|
| Programming Language | 🐍 Python |
| Machine Learning | 🤖 Scikit-learn |
| Algorithm | 📐 Linear Regression |
| Data Processing | 🐼 Pandas |
| Backend | ⚡ FastAPI |
| Server | 🚀 Uvicorn |
| Frontend | 🌐 HTML, CSS, JavaScript |
| Visualization | 📊 Plotly |
| Data Source | 📈 NSE Historical Stock Data |



## ✨ Key Features

- 🏢 Multiple company selection
- 📅 Future date selection
- 🔮 Future stock-price prediction
- 🤖 Linear Regression Machine Learning model
- 📊 Interactive stock-price graph
- 🗓️ Trading-day validation
- 📈 Historical price visualization
- 🔄 Dynamic prediction based on user input
- 💻 Simple localhost application

---

## 🧠 Why Linear Regression?

Linear Regression is suitable for this project because the output being predicted is a **continuous numerical value**, which is the stock's closing price.

For example:

- ₹725.40
- ₹731.25
- ₹745.80

Unlike classification algorithms that predict categories, Linear Regression predicts numerical values.

### Linear Regression vs Logistic Regression

| Linear Regression | Logistic Regression |
|---|---|
| Predicts continuous values | Mainly used for classification |
| Example: Stock Price | Example: Buy/Sell |
| Output can be ₹750.50 | Output is a class or probability |
| Used in this project | Not used in this project |

Therefore, **Linear Regression** is used because the required output is a numerical stock price.



## 📐 Mathematical Logic

The Linear Regression model uses the equation:

**y = mx + c**

Where:

- `y` = Predicted closing stock price
- `x` = Trading day number
- `m` = Slope of the regression line
- `c` = Intercept

The model finds the best values of `m` and `c` from the historical data.

For a future date, the system determines the corresponding trading-day number and uses the trained model to calculate the estimated stock price.

### Simple Example

Suppose the trained model produces:

m = 2.5  
c = 700

For a future trading day:

x = 20

Then:

y = mx + c

y = 2.5(20) + 700

y = 750

Therefore, the estimated closing price would be:

**₹750**



## 🔄 Example Workflow

🏢 Select Company  
↓  
📅 Select Future Date  
↓  
🗓️ Check Trading Day  
↓  
🔢 Calculate Trading Day Number  
↓  
🤖 Apply Linear Regression  
↓  
🔮 Generate Prediction  
↓  
💰 Display Predicted Price  
↓  
📊 Display Graph



## 🖥️ How to Run the Project

### 1️⃣ Clone the Repository

    git clone https://github.com/shweeta1015/stock-price-prediction-linear-regression.git

### 2️⃣ Open the Project Folder

    cd stock-price-prediction-linear-regression

### 3️⃣ Install Required Libraries

    pip install pandas scikit-learn fastapi uvicorn plotly

### 4️⃣ Run the Application

    python app.py

### 5️⃣ Open the Application

Open your browser and visit:

    http://127.0.0.1:5000



## ⚠️ Important Disclaimer

The stock prices generated by this application are **estimated predictions** produced by a Linear Regression model based on historical data.

Actual stock prices can be affected by many factors, including:

- Market conditions
- Company performance
- Economic events
- News
- Investor sentiment
- Government policies
- Global events

Therefore, the predicted value may differ significantly from the actual future market price.

This project is developed for **educational and academic purposes only** and should not be considered financial advice or a recommendation to buy or sell stocks.



## 🎓 Academic Purpose

This project demonstrates the practical implementation of:

- 📚 Machine Learning
- 📊 Data Analysis
- 📈 Linear Regression
- 🐍 Python Programming
- 🌐 Web Application Development
- 📉 Data Visualization

It demonstrates how historical real-world data can be used to build a simple Machine Learning-based stock-price prediction system.



## 👩‍💻 Author

**Shweta Thorat**

B.Tech Computer Science & Engineering

🎓 Machine Learning Project  
📈 Stock Price Prediction Using Linear Regression



## ⭐ Project Highlights

✨ Simple and easy-to-use interface  
✨ Multiple NSE companies supported  
✨ Dynamic future-date prediction  
✨ Historical stock-price analysis  
✨ Linear Regression-based prediction  
✨ Interactive graph visualization  
✨ Trading-day validation  
✨ Developed for academic and learning purposes
